"""Article body acquisition: HTML/markdown → clean Markdown + quality checks.

Every source funnels through here so the LLM always receives the same kind of
input: the article's main text as Markdown with headings, lists, tables and
fenced code blocks intact — and without navigation, ads, images, embeds or
invisible characters.

Why not BeautifulSoup ``get_text()``: it emits one line per inline element, so
"use <code>foo</code> to" becomes "use\\nfoo\\nto" and syntax-highlighted code
turns into one token per line. The LLM then has to guess at sentence and code
structure, which is where broken translations and invented code came from.
"""

from __future__ import annotations

import html
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import trafilatura
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from readability import Document

logger = logging.getLogger(__name__)

# Both libraries log every page they find unusual; that is expected here.
logging.getLogger("trafilatura").setLevel(logging.ERROR)
logging.getLogger("readability").setLevel(logging.ERROR)
logging.getLogger("readability.readability").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="bs4")

BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# --------------------------------------------------------------------------- #
# Markdown normalization
# --------------------------------------------------------------------------- #

_FENCED_BLOCK_RE = re.compile(r"(^|\n)[ \t]{0,3}(```|~~~)[^\n]*\n.*?\n[ \t]{0,3}\2[ \t]*(?=\n|$)", re.S)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
# Zero-width / bidi-control / soft-hyphen characters. Some sites pad text with
# thousands of these (watermarking); they only burn tokens and confuse models.
_INVISIBLE_RE = re.compile(r"[­᠎​-‏‪-‮⁠-⁤⁦-⁯﻿￹-￻]")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# The lookbehind keeps "![](url)" — kept images with an empty alt — intact
_EMPTY_LINK_RE = re.compile(r"(?<!!)\[\s*\]\([^)]*\)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HTML_HEADING_RE = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1\s*>", re.S | re.I)
_HTML_BLOCK_DROP_RE = re.compile(r"<(picture|video|audio|iframe|svg|script|style)\b.*?</\1\s*>", re.S | re.I)
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.I)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(\s[^<>]*)?/?>")

# Lines that are UI chrome when they appear on their own (never inside code).
_UI_LINES = frozenset(
    {
        "advertisement", "advertisements", "sponsored", "share", "share this", "share this article",
        "share this post", "tweet", "copy link", "link copied", "print", "subscribe", "sign in", "sign up",
        "log in", "login", "register", "menu", "search", "skip to content", "skip to main content",
        "back to top", "listen to this article", "read more", "learn more", "story text", "size", "width",
        "links", "prev story", "next story", "previous post", "next post", "previous article", "next article",
        "credit", "related", "related posts", "related articles", "more from", "follow", "facebook",
        "twitter", "linkedin", "reddit", "hacker news", "comments", "view comments", "show comments",
        "loading...", "loading", "close", "×",
    }
)


def _split_code(md: str) -> list[tuple[bool, str]]:
    """Split markdown into (is_code, text) segments around fenced code blocks."""
    segments: list[tuple[bool, str]] = []
    pos = 0
    for match in _FENCED_BLOCK_RE.finditer(md):
        start = match.start() + len(match.group(1))
        if start > pos:
            segments.append((False, md[pos:start]))
        segments.append((True, md[start:match.end()]))
        pos = match.end()
    if pos < len(md):
        segments.append((False, md[pos:]))
    return segments


def _map_prose(md: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to prose only; fenced code blocks and inline code are left untouched."""
    out = []
    for is_code, segment in _split_code(md):
        if is_code:
            out.append(segment)
            continue
        spans = []

        def _protect(m: re.Match) -> str:
            spans.append(m.group(0))
            return f"\x00{len(spans) - 1}\x00"

        # Keep the newlines that separate this prose from neighbouring code fences.
        core = segment.strip("\n")
        lead = segment[: len(segment) - len(segment.lstrip("\n"))]
        trail = segment[len(segment.rstrip("\n")):] if core else ""
        processed = fn(_INLINE_CODE_RE.sub(_protect, core))
        processed = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], processed)
        out.append(lead + processed + trail)
    return "".join(out)


def _strip_html_and_media(text: str, keep_images: bool = False) -> str:
    text = _HTML_COMMENT_RE.sub("", text)
    text = _HTML_BLOCK_DROP_RE.sub("", text)
    text = _HTML_HEADING_RE.sub(
        lambda m: "\n" + "#" * int(m.group(1)) + " " + _HTML_TAG_RE.sub("", m.group(2)).strip() + "\n", text
    )
    text = _HTML_BR_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub("", text)
    if not keep_images:
        text = _MD_IMAGE_RE.sub("", text)
    text = _EMPTY_LINK_RE.sub("", text)
    return text


def _drop_ui_lines(text: str) -> str:
    kept = []
    previous_block = None
    for block in re.split(r"\n{2,}", text):
        lines = [
            line for line in block.split("\n")
            if line.strip().rstrip(":").lower() not in _UI_LINES
        ]
        cleaned = "\n".join(lines).strip("\n")
        if not cleaned.strip():
            continue
        # Repeated captions/bylines ("Credit: X" twice in a row) add nothing.
        if cleaned == previous_block:
            continue
        kept.append(cleaned)
        previous_block = cleaned
    return "\n\n".join(kept)


def normalize_markdown(md: str, *, keep_images: bool = False) -> str:
    """Normalize extracted/source markdown into the shape the LLM receives.

    Images are dropped unless ``keep_images`` (README write-ups, which convert
    HTML images to Markdown beforehand — other HTML tags are still stripped).
    """
    if not md:
        return ""
    md = md.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    md = _INVISIBLE_RE.sub("", md)
    md = _map_prose(md, lambda text: _strip_html_and_media(text, keep_images))
    md = _map_prose(md, _drop_ui_lines)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# --------------------------------------------------------------------------- #
# Source-specific markdown cleanup
# --------------------------------------------------------------------------- #

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
_LIQUID_PAIRED_RE = re.compile(r"{%-?\s*(?:end\w+|raw|katex|details\b[^%]*|spoiler\b[^%]*|collapsible\b[^%]*)\s*-?%}")
_LIQUID_EMBED_RE = re.compile(r"{%-?\s*\w+[^%]*-?%}")


def clean_devto_markdown(md: str) -> str:
    """Dev.to body_markdown: drop front matter and Liquid embed tags ({% embed %}, {% youtube %}, ...)."""
    md = _FRONT_MATTER_RE.sub("", md or "")

    def _liquid(text: str) -> str:
        text = _LIQUID_PAIRED_RE.sub("", text)  # keep the inner text of {% details %}...{% enddetails %}
        return _LIQUID_EMBED_RE.sub("", text)

    return normalize_markdown(_map_prose(md, _liquid))


# --------------------------------------------------------------------------- #
# GitHub README images
# --------------------------------------------------------------------------- #
# Repository write-ups keep the README's screenshots, demos and diagrams. Image
# links are made absolute raw.githubusercontent.com URLs: relative paths and
# github.com/<owner>/<repo>/blob/... pages only resolve on github.com, while raw
# files are served with permissive CORS/CORP headers, so devport.kr can embed them.

_MD_IMAGE_PARTS_RE = re.compile(r"""!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+(?:"[^"]*"|'[^']*'))?\s*\)""")
_HTML_PICTURE_RE = re.compile(r"<picture\b[^>]*>(.*?)</picture\s*>", re.S | re.I)
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.I)
_HTML_SOURCE_RE = re.compile(r"<source\b[^>]*>", re.I)
_HTML_ATTR_RE = re.compile(r"""([a-zA-Z][\w:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")
_GITHUB_FILE_PATH_RE = re.compile(r"^/([^/]+)/([^/]+)/(?:blob|raw)/(.+)$")

# Badges, counters, contributor walls, star charts and generated headers are
# decoration, not content
_DECORATION_HOSTS = (
    "shields.io", "badgen.net", "badge.fury.io", "codecov.io", "coveralls.io", "circleci.com",
    "travis-ci.org", "travis-ci.com", "api.netlify.com", "readthedocs.org", "pepy.tech", "sonarcloud.io",
    "contrib.rocks", "star-history.com", "reporoster.com", "capsule-render.vercel.app", "komarev.com",
    "hits.seeyoufarm.com", "repobeats.axiom.co", "skillicons.dev", "github-readme-stats.vercel.app",
)
_DECORATION_PATH_RE = re.compile(r"badge|button|sponsor|devicon", re.I)
# Community, sponsor and deploy buttons, recognized by their alt text
_BUTTON_ALT_RE = re.compile(
    r"\b(discord|slack|telegram|twitter|wechat|follow us|sponsor|donate|buy me a coffee|ko-?fi|patreon"
    r"|visit|open in|deploy (?:to|with|on))\b",
    re.I,
)
# Only ";"-terminated references: html.unescape also decodes legacy entities
# without one, turning "?a=1&section=x" into "?a=1§ion=x"
_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);")
# Signed links copied from rendered READMEs expire within minutes
_EXPIRING_IMAGE_HOSTS = ("private-user-images.githubusercontent.com",)


def _unescape_entities(text: str) -> str:
    return _ENTITY_RE.sub(lambda m: html.unescape(m.group(0)), text)


def _html_attrs(tag: str) -> dict[str, str]:
    return {
        m.group(1).lower(): _unescape_entities(next(v for v in m.groups()[1:] if v is not None))
        for m in _HTML_ATTR_RE.finditer(tag)
    }


def _markdown_image(alt: str, src: str) -> str:
    alt = re.sub(r"[\[\]\s]+", " ", alt).strip()
    src = src.strip().replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"![{alt}]({src})" if src else ""


def _img_tag_to_markdown(tag: str) -> str:
    attrs = _html_attrs(tag)
    return _markdown_image(attrs.get("alt", ""), attrs.get("src", ""))


def _picture_to_markdown(match: re.Match) -> str:
    """<picture> → one Markdown image, preferring the dark-mode source (devport is dark-themed)."""
    inner = match.group(1)
    img = _HTML_IMG_RE.search(inner)
    alt = _html_attrs(img.group(0)).get("alt", "") if img else ""
    for source in _HTML_SOURCE_RE.findall(inner):
        attrs = _html_attrs(source)
        srcset = attrs.get("srcset", "").strip()
        if "dark" in attrs.get("media", "") and srcset:
            return _markdown_image(alt, srcset.split(",")[0].split()[0])
    return _img_tag_to_markdown(img.group(0)) if img else ""


def _raw_bases(download_url: str, readme_path: str) -> tuple[str, str]:
    """(README directory, repository root) as raw URLs, for resolving relative image paths."""
    if not download_url:
        return "", ""
    directory = download_url.rsplit("/", 1)[0] + "/"
    if readme_path and download_url.endswith(readme_path):
        return directory, download_url[: -len(readme_path)]
    return directory, directory


def _readme_image_url(src: str, raw_dir: str, raw_root: str) -> Optional[str]:
    """Absolute, embeddable URL for a README image, or None if the image should be dropped."""
    url, _, fragment = _unescape_entities(src).strip().partition("#")
    if not url or fragment == "gh-light-mode-only":
        return None  # its #gh-dark-mode-only twin is kept instead
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):  # repository-root relative on GitHub
        url = urljoin(raw_root, url.lstrip("/")) if raw_root else ""
    elif not urlparse(url).scheme:
        url = urljoin(raw_dir, url) if raw_dir else ""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if parsed.scheme not in ("http", "https") or host in _EXPIRING_IMAGE_HOSTS:
        return None
    if any(host == h or host.endswith("." + h) for h in _DECORATION_HOSTS) or _DECORATION_PATH_RE.search(parsed.path):
        return None
    if host == "github.com":
        file_match = _GITHUB_FILE_PATH_RE.match(parsed.path)
        if file_match:  # blob/raw page → the file itself
            owner, repo, rest = file_match.groups()
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{rest}"
    return url


def _rewrite_readme_image(match: re.Match, raw_dir: str, raw_root: str) -> str:
    alt, src = match.group(1), match.group(2)
    if _BUTTON_ALT_RE.search(alt):
        return ""
    url = _readme_image_url(src, raw_dir, raw_root)
    return _markdown_image(alt, url) if url else ""


def normalize_readme_markdown(md: str, download_url: str = "", readme_path: str = "") -> str:
    """README → the Markdown the repo write-up model receives, images included.

    HTML <picture>/<img> become Markdown images, every image URL is made
    absolute against the README's raw URL (``download_url`` from the GitHub
    contents API), and badges, community buttons and light-mode twins are dropped.
    """
    raw_dir, raw_root = _raw_bases(download_url, readme_path)

    def _images(text: str) -> str:
        text = _HTML_COMMENT_RE.sub("", text)  # commented-out images must stay out
        text = _HTML_PICTURE_RE.sub(_picture_to_markdown, text)
        text = _HTML_IMG_RE.sub(lambda m: _img_tag_to_markdown(m.group(0)), text)
        return _MD_IMAGE_PARTS_RE.sub(lambda m: _rewrite_readme_image(m, raw_dir, raw_root), text)

    return normalize_markdown(_map_prose(md or "", _images), keep_images=True)


def drop_unknown_images(text: str, source_md: str) -> str:
    """Keep only Markdown images whose URL appears in ``source_md``; invented or altered links go."""
    allowed = {m.group(2) for m in _MD_IMAGE_PARTS_RE.finditer(source_md or "")}

    def _filter(prose: str) -> str:
        prose = _HTML_IMG_RE.sub("", prose)
        prose = _MD_IMAGE_PARTS_RE.sub(lambda m: m.group(0) if m.group(2) in allowed else "", prose)
        return _EMPTY_LINK_RE.sub("", prose)

    return re.sub(r"\n{3,}", "\n\n", _map_prose(text or "", _filter)).strip()


# --------------------------------------------------------------------------- #
# HTML → markdown
# --------------------------------------------------------------------------- #

_DROP_TAGS = [
    "script", "style", "noscript", "template", "iframe", "svg", "canvas", "button", "form", "input",
    "select", "textarea", "nav", "aside", "footer", "img", "picture", "video", "audio", "source", "dialog",
]
_BOILERPLATE_ATTR_RE = re.compile(
    r"(share|social|related|recommend|popular|most-?read|trending|newsletter|subscribe|signup|sign-up|"
    r"comment|promo|advert|sponsor|sidebar|breadcrumb|cookie|consent|popup|modal|paywall|author-?bio|"
    r"byline|toolbar|pagination|prev-next|read-?more|tag-?list|footer|header|masthead)",
    re.I,
)
_CONTAINER_SELECTORS = (
    '[itemprop="articleBody"]', "article", "main", '[role="main"]', ".post-content", ".entry-content",
    ".article-content", ".article-body", ".markdown-body", ".post-body", "#content",
)
_LANG_CLASS_RE = re.compile(r"(?:language|lang|highlight-source)-([\w+#.-]+)")


def _code_language(el) -> Optional[str]:
    for node in (el, el.find("code")):
        if node is None:
            continue
        for cls in node.get("class") or []:
            match = _LANG_CLASS_RE.match(cls)
            if match:
                return match.group(1).lower()
    return None


def _to_markdown(fragment) -> str:
    return MarkdownConverter(
        heading_style="ATX",
        bullets="-",
        strip=["a"],
        escape_asterisks=False,
        escape_underscores=False,
        escape_misc=False,
        code_language_callback=_code_language,
    ).convert_soup(fragment)


def _strip_boilerplate(root) -> None:
    """Remove chrome from a content subtree, never a block holding most of its text."""
    for tag in root.find_all(_DROP_TAGS):
        tag.decompose()
    total = len(root.get_text(" ", strip=True)) or 1
    for tag in list(root.find_all(True)):
        if tag.decomposed or not tag.attrs:
            continue
        tokens = list(tag.get("class") or [])
        if tag.get("id"):
            tokens.append(tag.get("id"))
        if not any(isinstance(t, str) and _BOILERPLATE_ATTR_RE.search(t) for t in tokens):
            continue
        if len(tag.get_text(" ", strip=True)) < 0.3 * total:
            tag.decompose()


def html_fragment_to_markdown(html: str) -> str:
    """Convert a small trusted HTML fragment (HN post text, Dev.to body_html) to markdown."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    return normalize_markdown(_to_markdown(soup))


def _trafilatura(html: str, url: str, favor_recall: bool) -> str:
    try:
        return trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_formatting=True,
            include_tables=True,
            include_images=False,
            include_links=False,
            include_comments=False,
            favor_recall=favor_recall,
            favor_precision=not favor_recall,
            deduplicate=True,
        ) or ""
    except Exception as e:
        logger.debug(f"trafilatura failed for {url}: {e}")
        return ""


def _readability(html: str, url: str) -> str:
    try:
        soup = BeautifulSoup(Document(html, url=url).summary(html_partial=True), "lxml")
    except Exception as e:
        logger.debug(f"readability failed for {url}: {e}")
        return ""
    _strip_boilerplate(soup)
    return _to_markdown(soup)


def _main_container(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    best, best_len = None, 0
    for selector in _CONTAINER_SELECTORS:
        for node in soup.select(selector):
            length = len(node.get_text(" ", strip=True))
            if length > best_len:
                best, best_len = node, length
    if best is None:
        return ""
    _strip_boilerplate(best)
    return _to_markdown(best)


def prose_chars(md: str) -> int:
    """Characters in real sentences or code — menus and button labels don't count."""
    total = 0
    for is_code, segment in _split_code(md or ""):
        if is_code:
            total += len(segment)
            continue
        total += sum(
            len(line) for line in segment.split("\n")
            if len(line.split()) >= 8 or len(line.strip()) >= 80
        )
    return total


def extract_main_content(html: str, url: str) -> tuple[str, str]:
    """Extract the main article body as markdown. Returns (markdown, extractor_name).

    trafilatura is the most precise extractor on average, but it sometimes picks
    the wrong block (LessWrong) or stops at an in-article ad break (Ars Technica).
    readability and a plain <article>/<main> container conversion are used as
    second opinions and only win when they recover clearly more real prose.
    """
    if not html:
        return "", "none"

    primary = normalize_markdown(_trafilatura(html, url, favor_recall=True))
    precise = normalize_markdown(_trafilatura(html, url, favor_recall=False))
    if prose_chars(precise) > prose_chars(primary) * 1.3:
        primary = precise

    readable = normalize_markdown(_readability(html, url))
    container = normalize_markdown(_main_container(html))

    p_score, r_score, c_score = prose_chars(primary), prose_chars(readable), prose_chars(container)
    best, method = primary, "trafilatura"
    if r_score > p_score * 1.6 and r_score > 500:
        best, method = readable, "readability"
    if c_score > max(p_score, r_score) * 2.5 and c_score > 1500:
        best, method = container, "container"
    return best, method


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #

_BLOCKED_PHRASES: tuple[tuple[str, str], ...] = (
    ("enable javascript", "js_required"),
    ("requires javascript", "js_required"),
    ("javascript is disabled", "js_required"),
    ("javascript to run this app", "js_required"),
    ("turn on javascript", "js_required"),
    ("just a moment", "bot_challenge"),
    ("checking your browser", "bot_challenge"),
    ("verify you are human", "bot_challenge"),
    ("are you a robot", "bot_challenge"),
    ("attention required", "bot_challenge"),
    ("unusual traffic", "bot_challenge"),
    ("access denied", "blocked"),
    ("request blocked", "blocked"),
    ("403 forbidden", "blocked"),
    ("subscribe to continue", "paywall"),
    ("subscribe to read", "paywall"),
    ("to continue reading", "paywall"),
    ("this post is for paid subscribers", "paywall"),
    ("member-only story", "paywall"),
    ("already a subscriber", "paywall"),
    ("sign in to continue", "login_wall"),
    ("log in to continue", "login_wall"),
    ("page not found", "not_found"),
    ("404 not found", "not_found"),
    ("this page could not be found", "not_found"),
)


@dataclass(slots=True)
class ContentCheck:
    ok: bool
    reason: str = ""


def assess_content(md: str, min_chars: int) -> ContentCheck:
    """Cheap deterministic check that ``md`` is a readable article body."""
    text = (md or "").strip()
    if not text:
        return ContentCheck(False, "empty")
    length = len(text)
    if length < 3000:
        head = text[:3000].lower()
        for phrase, reason in _BLOCKED_PHRASES:
            if phrase in head:
                return ContentCheck(False, reason)
    if length < min_chars:
        return ContentCheck(False, f"too_short:{length}")
    if prose_chars(text) < min_chars * 0.5:
        return ContentCheck(False, "mostly_fragments")
    return ContentCheck(True)


def truncate_markdown(md: str, max_chars: int) -> tuple[str, bool]:
    """Cut ``md`` at a block boundary at or before ``max_chars``, never mid code block."""
    if len(md) <= max_chars:
        return md, False
    out: list[str] = []
    size = 0
    for is_code, segment in _split_code(md):
        if size + len(segment) <= max_chars:
            out.append(segment)
            size += len(segment)
            continue
        if not is_code:
            remaining = max_chars - size
            cut = segment.rfind("\n\n", 0, remaining)
            if cut < remaining * 0.8:
                # Long lists have few blank lines; don't throw away most of the budget.
                cut = max(cut, segment.rfind("\n", 0, remaining))
            out.append(segment[: cut if cut > 0 else remaining])
        break
    result = "".join(out).rstrip()
    return (result or md[:max_chars]), True


# --------------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------------- #

_TRACKING_PARAM_RE = re.compile(
    r"^(utm_[a-z_]+|fbclid|gclid|dclid|gbraid|wbraid|msclkid|mc_cid|mc_eid|igshid|yclid|_hsenc|_hsmi|"
    r"mkt_tok|ref_src|ref_url)$",
    re.I,
)
_NO_ARTICLE_HOSTS = (
    "youtube.com", "youtu.be", "vimeo.com", "twitch.tv", "tiktok.com", "v.redd.it", "i.redd.it",
    "streamable.com", "x.com", "twitter.com", "bsky.app", "instagram.com", "facebook.com",
    "threads.net", "i.imgur.com", "imgur.com",
)
_BINARY_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".mov", ".webm", ".mp3", ".zip",
    ".tar.gz", ".dmg", ".exe",
)
_GITHUB_PATH_RE = re.compile(r"^/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/(tree|blob)/([^/]+)(/.*)?)?/?$")
_GITHUB_RESERVED_OWNERS = frozenset(
    {"features", "about", "orgs", "sponsors", "settings", "marketplace", "topics", "collections",
     "trending", "events", "enterprise", "pricing", "security", "login", "join", "explore", "apps"}
)


def canonicalize_url(url: str) -> str:
    """Drop tracking query params and anchors so the same link dedups across sources."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if not parsed.scheme or not parsed.netloc:
        return url
    query = parse_qsl(parsed.query, keep_blank_values=True)
    kept = [(k, v) for k, v in query if not _TRACKING_PARAM_RE.match(k)]
    # Hash routing (#/path, #!/path) is part of the address on some SPAs.
    keep_fragment = parsed.fragment.startswith(("/", "!"))
    if len(kept) == len(query) and (keep_fragment or not parsed.fragment):
        return url
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        urlencode(kept, doseq=True) if len(kept) != len(query) else parsed.query,
        parsed.fragment if keep_fragment else "",
    ))


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def unsupported_reason(url: str) -> Optional[str]:
    """Reason a URL cannot yield an article body (video, social post, binary), else None."""
    try:
        host = _host(url)
        path = urlparse(url).path.lower()
    except ValueError:
        return "invalid_url"
    if any(host == h or host.endswith("." + h) for h in _NO_ARTICLE_HOSTS):
        return f"no_article_host:{host}"
    if path.endswith(_BINARY_EXTENSIONS):
        return "binary_file"
    return None


@dataclass(slots=True)
class GitHubRef:
    owner: str
    repo: str
    ref: Optional[str] = None
    path: Optional[str] = None  # set for blob/tree URLs pointing at a file


def parse_github_url(url: str) -> Optional[GitHubRef]:
    """Recognize github.com repo roots and markdown file links (README via API is far cleaner)."""
    try:
        if _host(url) != "github.com":
            return None
        match = _GITHUB_PATH_RE.match(urlparse(url).path)
    except ValueError:
        return None
    if not match:
        return None
    owner, repo, kind, ref, path = match.groups()
    if owner.lower() in _GITHUB_RESERVED_OWNERS:
        return None
    if kind == "blob":
        if not path or not path.lower().endswith((".md", ".markdown", ".mdx", ".rst", ".txt")):
            return None
        return GitHubRef(owner, repo, ref, path.lstrip("/"))
    if kind == "tree" and path:
        return None  # sub-directory listing — let the HTML path handle it
    return GitHubRef(owner, repo, ref)
