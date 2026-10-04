"""Base crawler class with common functionality"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging
import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from app.config.settings import settings
from app.crawlers.content import (
    BROWSER_HEADERS,
    GitHubRef,
    assess_content,
    extract_main_content,
    normalize_markdown,
    parse_github_url,
    prose_chars,
    unsupported_reason,
)
from app.utils import deadline

logger = logging.getLogger(__name__)

# HTTP status codes that warrant a retry (transient server errors / rate limits)
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Parse at most this much HTML; a few pages are tens of MB of inline data.
_MAX_HTML_CHARS = 3_000_000

# Skip the Playwright fallback when the Lambda has less time than this left.
_MIN_SECONDS_FOR_BROWSER = 180


@dataclass(slots=True)
class FetchedContent:
    """Result of fetching one article URL."""

    markdown: str
    method: str  # github_api | httpx:<extractor> | playwright:<extractor> | skipped
    issue: str = ""  # why the content is unusable; empty when it passed the quality gate


class _HttpRetryableError(Exception):
    """Internal signal raised inside _retryable_http_request so tenacity retries on 429/5xx."""


class RawArticle:
    """
    Raw article data before processing

    This is the intermediate format before converting to Article model
    """
    def __init__(
        self,
        title_en: str,
        url: str,
        source: str,
        published_at: datetime,
        external_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        content: Optional[str] = None,
        stars: Optional[int] = None,
        comments: Optional[int] = None,
        upvotes: Optional[int] = None,
        read_time: Optional[str] = None,
        language: Optional[str] = None,
        raw_data: Optional[Dict[str, Any]] = None
    ):
        self.title_en = title_en
        self.url = url
        self.source = source
        self.published_at = published_at
        self.external_id = external_id  # Optional: source's actual ID
        self.tags = tags or []
        self.content = content
        self.stars = stars
        self.comments = comments
        self.upvotes = upvotes
        self.read_time = read_time
        self.language = language
        self.raw_data = raw_data or {}

    def __repr__(self):
        return f"<RawArticle {self.source}: {self.title_en[:50]}>"


class BaseCrawler(ABC):
    """
    Base crawler class

    All source-specific crawlers inherit from this class
    """

    def __init__(self, known_url_filter: Optional[Callable[[List[str]], set]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.user_agent = settings.USER_AGENT
        self.delay = settings.CRAWL_DELAY_SECONDS
        # Returns the subset of URLs already saved, so their (slow) content
        # fetch can be skipped — dedup used to happen only after fetching.
        self.known_url_filter = known_url_filter
        self._playwright_page = None
        self._playwright_page_lock = asyncio.Lock()
        self._browser = None
        self._pw = None
        self._browser_attempted = False
        self._browser_lock = asyncio.Lock()
        self._http_sem = asyncio.Semaphore(settings.CONTENT_FETCH_CONCURRENCY)

    async def _diagnose_chromium_failure(self):
        """
        One-shot diagnostic: locate the Playwright Chromium binary and run it
        with --version + --no-sandbox + --headless. Capture stderr so we can
        see the actual reason Chromium is dying in Lambda (missing lib name,
        segfault, /dev/shm error, etc).
        """
        import asyncio as _asyncio
        import os
        import glob

        try:
            browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/ms-playwright")
            candidates = (
                glob.glob(f"{browsers_path}/chromium-*/chrome-linux/headless_shell")
                + glob.glob(f"{browsers_path}/chromium_headless_shell-*/chrome-linux/headless_shell")
                + glob.glob(f"{browsers_path}/chromium-*/chrome-linux/chrome")
            )
            logger.error(f"DIAG: PLAYWRIGHT_BROWSERS_PATH={browsers_path}, candidates={candidates}")
            if not candidates:
                logger.error(f"DIAG: no chromium binary found under {browsers_path}")
                # List what IS there
                if os.path.isdir(browsers_path):
                    for entry in os.listdir(browsers_path):
                        logger.error(f"DIAG:   {browsers_path}/{entry}")
                return

            chrome_bin = candidates[0]
            logger.error(f"DIAG: testing {chrome_bin}")

            # Run with --version (lightweight) and capture stderr
            proc = await _asyncio.create_subprocess_exec(
                chrome_bin,
                "--version",
                "--no-sandbox",
                "--headless",
                "--disable-gpu",
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=10.0)
                logger.error(
                    f"DIAG: chromium exit={proc.returncode}, "
                    f"stdout={stdout.decode(errors='replace')[:500]!r}, "
                    f"stderr={stderr.decode(errors='replace')[:2000]!r}"
                )
            except _asyncio.TimeoutError:
                proc.kill()
                logger.error("DIAG: chromium --version hung for 10s")

            # Also run ldd to find any missing libs
            try:
                ldd = await _asyncio.create_subprocess_exec(
                    "ldd", chrome_bin,
                    stdout=_asyncio.subprocess.PIPE,
                    stderr=_asyncio.subprocess.PIPE,
                )
                stdout, _ = await _asyncio.wait_for(ldd.communicate(), timeout=5.0)
                missing = [
                    line for line in stdout.decode(errors="replace").splitlines()
                    if "not found" in line
                ]
                if missing:
                    logger.error(f"DIAG: missing libs:\n" + "\n".join(missing))
                else:
                    logger.error("DIAG: ldd reports all libs resolved")
            except Exception as e:
                logger.error(f"DIAG: ldd failed: {e}")

        except Exception as e:
            logger.error(f"DIAG: diagnostic itself failed: {e}", exc_info=True)

    async def _launch_browser(self):
        """
        Launch Playwright Chromium. Returns (browser, playwright) or (None, None).

        On Lambda the browser handle can be returned successfully even though the
        underlying Chromium subprocess died seconds later (OOM, /tmp full, missing
        lib). We do a smoke test by opening + closing one page; if that fails the
        whole crawl falls back to httpx-only instead of issuing 80 doomed page
        opens that all error with "Target page, context or browser has been closed".
        """
        try:
            from playwright.async_api import async_playwright
            pw = await async_playwright().start()
            # Force full chrome binary instead of chrome-headless-shell.
            # headless_shell is a stripped Chromium that has known broken
            # behavior under Lambda's restricted runtime (page targets fail
            # to instantiate). The full chrome binary is heavier but more
            # tolerant of unusual environments.
            import glob as _glob
            chrome_candidates = (
                _glob.glob("/ms-playwright/chromium-*/chrome-linux/chrome")
                + _glob.glob("/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell")
            )
            chrome_path = chrome_candidates[0] if chrome_candidates else None

            browser = await pw.chromium.launch(
                executable_path=chrome_path,
                headless=settings.PLAYWRIGHT_HEADLESS,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-gpu",
                    "--disable-accelerated-2d-canvas",
                    "--disable-dev-shm-usage",
                    # CRITICAL for Lambda: Chromium's multi-process model breaks
                    # under Lambda's PID namespace. --single-process collapses
                    # everything into the browser process. Combined with
                    # PLAYWRIGHT_CONCURRENCY=1 to avoid the multi-target race
                    # that single-process Chromium can't handle.
                    "--single-process",
                    "--no-zygote",
                    "--in-process-gpu",
                    "--disable-background-networking",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-features=AudioServiceOutOfProcess,IsolateOrigins,site-per-process",
                    "--mute-audio",
                    "--no-first-run",
                    "--no-default-browser-check",
                    f"--user-agent={settings.PLAYWRIGHT_USER_AGENT}",
                ],
            )

            # Smoke test: prove Chromium can create a page target, then keep
            # that page open for reuse. In Lambda-compatible --single-process
            # mode, Chromium reliably supports navigating one existing page but
            # fails when asked to create a second page target.
            try:
                self._playwright_page = await browser.new_page()
                await self._configure_playwright_page(self._playwright_page)
                await self._playwright_page.goto(
                    "data:text/html,<title>smoke</title>",
                    wait_until="domcontentloaded",
                    timeout=5000,
                )
            except Exception as smoke_err:
                logger.error(
                    f"Browser launched but smoke test failed (Chromium dead): {smoke_err}. "
                    f"Falling back to httpx-only for this crawl."
                )
                # Diagnostic: run the chromium binary directly and capture stderr
                # so we can see WHY it's dying (missing lib, segfault, etc).
                await self._diagnose_chromium_failure()
                try:
                    await browser.close()
                except Exception:
                    pass
                await pw.stop()
                return None, None

            logger.info("Playwright browser launched and smoke-tested successfully")
            return browser, pw
        except Exception as e:
            logger.warning(f"Playwright not available, falling back to httpx-only: {e}")
            return None, None

    async def _get_browser(self):
        """Launch Chromium on first use only — most pages never need it."""
        async with self._browser_lock:
            if not self._browser_attempted:
                self._browser_attempted = True
                self._browser, self._pw = await self._launch_browser()
            return self._browser

    async def close_browser(self) -> None:
        browser, pw = self._browser, self._pw
        self._browser = self._pw = self._playwright_page = None
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
        if pw:
            try:
                await pw.stop()
            except Exception:
                pass

    def drop_known(self, articles: List["RawArticle"]) -> List["RawArticle"]:
        """Drop articles whose URL is already saved, before any content fetching."""
        if not self.known_url_filter or not articles:
            return articles
        try:
            known = self.known_url_filter([a.url for a in articles])
        except Exception as e:
            self.logger.warning(f"Known-URL lookup failed, fetching everything: {e}")
            return articles
        if known:
            self.logger.info(f"Skipping {len(known)} already-saved URLs before content fetch")
        return [a for a in articles if a.url not in known]

    @abstractmethod
    async def crawl(self) -> List[RawArticle]:
        """
        Crawl the source and return raw articles

        Returns:
            List of RawArticle objects
        """
        pass

    @abstractmethod
    def should_skip(self, article: RawArticle) -> bool:
        """
        Determine if article should be skipped

        Args:
            article: RawArticle to check

        Returns:
            True if article should be skipped, False otherwise
        """
        pass

    def log_start(self):
        """Log crawl start"""
        self.logger.info(f"Starting {self.__class__.__name__}")

    def log_end(self, count: int):
        """Log crawl end with count"""
        self.logger.info(f"Finished {self.__class__.__name__}: {count} articles")

    def log_error(self, error: Exception):
        """Log error"""
        self.logger.error(f"Error in {self.__class__.__name__}: {str(error)}", exc_info=True)

    @staticmethod
    async def _retryable_http_request(
        method: str,
        url: str,
        *,
        client: httpx.AsyncClient,
        max_retries: int = None,
        backoff_base: float = None,
        backoff_max: float = None,
        **httpx_kwargs,
    ) -> httpx.Response:
        """
        Make an HTTP request with exponential backoff retry on transient failures.

        Retries on: httpx.TransportError (includes timeouts/network errors), HTTP 429/500/502/503/504.
        For HTTP 429: reads Retry-After header and waits before retrying.
        Does NOT retry: HTTP 400/401/403/404/422 (permanent failures).

        Returns the response on success.
        Raises _HttpRetryableError if all retries are exhausted on 429/5xx.
        Raises httpx.HTTPStatusError immediately for permanent client errors (non-retryable 4xx).
        """
        _max = max_retries if max_retries is not None else getattr(settings, "CRAWLER_HTTP_MAX_RETRIES", 3)
        _base = backoff_base if backoff_base is not None else getattr(settings, "CRAWLER_HTTP_BACKOFF_BASE_SECONDS", 1.0)
        _max_wait = backoff_max if backoff_max is not None else getattr(settings, "CRAWLER_HTTP_BACKOFF_MAX_SECONDS", 10.0)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_max),
            wait=wait_exponential(multiplier=_base, max=_max_wait),
            retry=retry_if_exception_type((_HttpRetryableError, httpx.TransportError)),
            reraise=True,
        ):
            with attempt:
                response = await client.request(method, url, **httpx_kwargs)

                if response.status_code in _RETRYABLE_STATUS_CODES:
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_secs = max(float(retry_after), 0.0)
                                logger.warning(f"Rate limited (429) by {url}, waiting {wait_secs:.1f}s per Retry-After")
                                await asyncio.sleep(wait_secs)
                            except ValueError:
                                pass
                        else:
                            logger.warning(f"Rate limited (429) by {url}, will use exponential backoff")
                    else:
                        logger.warning(f"HTTP {response.status_code} from {url}, will retry")
                    raise _HttpRetryableError(f"HTTP {response.status_code} for {url}")

                response.raise_for_status()
                return response

        # Unreachable (tenacity reraises), but satisfies the type checker
        raise _HttpRetryableError(f"All retries exhausted for {url}")

    async def fetch_article_content(self, client: httpx.AsyncClient, url: str) -> FetchedContent:
        """Fetch an article body as clean markdown.

        Order: skip URLs that never have an article body (video, social posts,
        binaries) → GitHub API for repo/markdown links → plain HTTP + extraction
        → Playwright render only when the static result fails the quality gate
        (SPA shells, bot walls, near-empty extractions).

        ``client`` must not carry source-specific auth headers: it talks to
        arbitrary third-party sites.
        """
        reason = unsupported_reason(url)
        if reason:
            return FetchedContent("", "skipped", reason)

        github = parse_github_url(url)
        if github:
            markdown = await self.fetch_github_markdown(client, github)
            if markdown:
                return FetchedContent(markdown, "github_api")

        min_chars = settings.MIN_ARTICLE_CONTENT_CHARS
        markdown, method, status, content_type = await self._fetch_static_markdown(client, url)
        check = assess_content(markdown, min_chars)
        if check.ok:
            return FetchedContent(markdown, method)
        if status in (404, 410):
            return FetchedContent(markdown, method, f"http_{status}")
        if content_type and "html" not in content_type:
            return FetchedContent("", method, f"not_html:{content_type.split(';')[0]}")
        if deadline.remaining() < _MIN_SECONDS_FOR_BROWSER:
            return FetchedContent(markdown, method, check.reason)

        browser = await self._get_browser()
        if browser is None:
            return FetchedContent(markdown, method, check.reason)

        html = await self.fetch_rendered_html(browser, url, timeout_ms=settings.PLAYWRIGHT_TIMEOUT_MS)
        rendered, extractor = ("", "none")
        if html:
            rendered, extractor = await asyncio.to_thread(extract_main_content, html[:_MAX_HTML_CHARS], url)
        rendered_check = assess_content(rendered, min_chars)
        if rendered_check.ok or prose_chars(rendered) > prose_chars(markdown):
            return FetchedContent(rendered, f"playwright:{extractor}", rendered_check.reason)
        return FetchedContent(markdown, method, check.reason)

    async def _fetch_static_markdown(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[str, str, Optional[int], str]:
        """GET a page with browser-like headers and extract its main content.

        Returns (markdown, method, http_status, content_type). Never raises.
        """
        headers = {**BROWSER_HEADERS, "User-Agent": settings.PLAYWRIGHT_USER_AGENT}
        try:
            async with self._http_sem:
                response = await client.get(url, headers=headers, follow_redirects=True, timeout=15.0)
        except Exception as e:
            logger.debug(f"Static fetch failed for {url}: {e}")
            return "", "httpx", None, ""

        content_type = response.headers.get("content-type", "").lower()
        if response.status_code >= 400 or "html" not in content_type:
            return "", "httpx", response.status_code, content_type

        markdown, extractor = await asyncio.to_thread(
            extract_main_content, response.text[:_MAX_HTML_CHARS], str(response.url)
        )
        return markdown, f"httpx:{extractor}", response.status_code, content_type

    @staticmethod
    async def fetch_github_markdown(client: httpx.AsyncClient, ref: GitHubRef) -> str:
        """Raw README (or linked markdown file) via the GitHub API — far cleaner than the HTML page."""
        headers = {
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": settings.USER_AGENT,
        }
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"
        base = f"https://api.github.com/repos/{ref.owner}/{ref.repo}"
        endpoint = f"{base}/contents/{ref.path}" if ref.path else f"{base}/readme"
        try:
            response = await client.get(
                endpoint,
                headers=headers,
                params={"ref": ref.ref} if ref.ref else None,
                timeout=15.0,
            )
        except Exception as e:
            logger.debug(f"GitHub API fetch failed for {endpoint}: {e}")
            return ""
        if response.status_code != 200:
            logger.debug(f"GitHub API {response.status_code} for {endpoint}")
            return ""
        return normalize_markdown(response.text)

    async def fetch_rendered_html(self, browser, url: str, timeout_ms: int = 30000) -> str:
        """Render a URL with Playwright (JS executed) and return the resulting HTML.

        Reuses the single page created during the browser smoke test (Lambda's
        --single-process Chromium cannot reliably open a second page target).
        Returns an empty string on any failure.
        """
        if not url:
            return ""

        # Fail fast if Chromium has died — avoids a queue of doomed navigations
        # all hitting "Target page, context or browser has been closed".
        if not browser.is_connected():
            return ""

        async with self._playwright_page_lock:
            page = None
            try:
                page = self._playwright_page
                if page is None or page.is_closed():
                    page = await browser.new_page()
                    await self._configure_playwright_page(page)
                    self._playwright_page = page

                await self._reset_playwright_page(page)

                # Use "commit" for the navigation itself. Some sites keep
                # DOMContentLoaded hostage behind redirects, H2 errors, ad
                # scripts, or SPA boot work; with a reused single page that can
                # leave a stale navigation active and interrupt the next URL.
                await page.goto(url, wait_until="commit", timeout=timeout_ms)

                try:
                    await page.wait_for_load_state(
                        "domcontentloaded",
                        timeout=min(timeout_ms, 5000),
                    )
                except Exception:
                    pass

                await self._wait_for_text_to_settle(page, settings.PLAYWRIGHT_SETTLE_MS)
                return await page.content()

            except Exception as e:
                logger.warning(f"Playwright failed for {url}: {e}")
                return ""
            finally:
                if page is not None and not page.is_closed():
                    await self._reset_playwright_page(page)

    @staticmethod
    async def _wait_for_text_to_settle(page, max_ms: int) -> None:
        """Wait until client-side rendering stops adding text, up to ``max_ms``.

        A fixed short sleep returned SPA shells ("Loading…") before the article
        had rendered; a long fixed sleep wastes time on static pages.
        """
        previous, stable_polls, waited = -1, 0, 0
        while waited < max_ms:
            await page.wait_for_timeout(500)
            waited += 500
            try:
                current = await page.evaluate(
                    "() => document.body ? document.body.innerText.length : 0"
                )
            except Exception:
                # Execution context replaced by a client-side redirect; keep waiting.
                previous, stable_polls = -1, 0
                continue
            if current > 0 and current == previous:
                stable_polls += 1
                if stable_polls >= 2:
                    return
            else:
                stable_polls = 0
            previous = current

    @staticmethod
    async def _configure_playwright_page(page) -> None:
        """Apply browser-page defaults that reduce CDN/headless rejects."""
        await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

    @staticmethod
    async def _reset_playwright_page(page) -> None:
        """Stop any in-flight navigation and return the reused page to blank."""
        try:
            await page.evaluate("window.stop()")
        except Exception:
            pass
        try:
            await page.goto("about:blank", wait_until="commit", timeout=3000)
        except Exception:
            pass

    @staticmethod
    async def send_discord_webhook(source_name: str, failed_articles: list[dict]) -> None:
        """
        Send failed article info to Discord webhook.

        Args:
            source_name: crawler name (e.g. "HackerNews", "Devto")
            failed_articles: list of dicts with keys: title, url, discussion_url, upvotes, comments
        """
        webhook_url = settings.DISCORD_WEBHOOK_URL
        if not webhook_url or not failed_articles:
            return

        # Build individual entry strings
        entries = []
        for i, art in enumerate(failed_articles, 1):
            entry = f"\n**{i}. {art['title'][:80]}**"
            entry += f"\n🔗 {art['url']}"
            if art.get("discussion_url"):
                entry += f"\n💬 {art['discussion_url']}"
            if art.get("upvotes") is not None or art.get("comments") is not None:
                stats = []
                if art.get("upvotes") is not None:
                    stats.append(f"{art['upvotes']} points")
                if art.get("comments") is not None:
                    stats.append(f"{art['comments']} comments")
                entry += f"\n📊 {' | '.join(stats)}"
            entries.append(entry)

        # Split into multiple messages to stay under Discord 2000 char limit
        header = (
            f"🔴 **Failed Content Fetches — {source_name}**\n"
            f"{len(failed_articles)} articles with insufficient content for summarization\n"
        )
        messages = []
        current = header
        for entry in entries:
            if len(current) + len(entry) > 1900:
                messages.append(current)
                current = f"🔴 **...continued ({source_name})**\n"
            current += entry
        messages.append(current)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for msg in messages:
                    resp = await client.post(webhook_url, json={"content": msg})
                    resp.raise_for_status()
                logger.info(f"Sent Discord webhook: {len(failed_articles)} failed articles from {source_name} ({len(messages)} messages)")
        except Exception as e:
            logger.warning(f"Failed to send Discord webhook: {e}")
