"""Base crawler class with common functionality"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging
import re
import httpx
from bs4 import BeautifulSoup
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from app.config.settings import settings

logger = logging.getLogger(__name__)

MIN_CONTENT_LENGTH = 6000

# HTTP status codes that warrant a retry (transient server errors / rate limits)
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


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

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.user_agent = settings.USER_AGENT
        self.delay = settings.CRAWL_DELAY_SECONDS

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
            browser = await pw.chromium.launch(
                headless=settings.PLAYWRIGHT_HEADLESS,
                args=[
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-features=AudioServiceOutOfProcess,IsolateOrigins,site-per-process",
                    "--disable-background-networking",
                    "--disable-renderer-backgrounding",
                    "--disable-backgrounding-occluded-windows",
                    "--mute-audio",
                ],
            )

            # Smoke test: prove Chromium subprocess is actually alive
            try:
                test_page = await browser.new_page()
                await test_page.close()
            except Exception as smoke_err:
                logger.error(
                    f"Browser launched but smoke test failed (Chromium dead): {smoke_err}. "
                    f"Falling back to httpx-only for this crawl."
                )
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

    @staticmethod
    async def fetch_url_content(
        client: httpx.AsyncClient,
        url: str,
        user_agent: str = None,
        max_chars: int = 15000,
    ) -> str:
        """
        Fetch and extract readable text from an article URL.

        Uses BeautifulSoup to extract main content from HTML pages.
        Returns empty string on any failure (non-blocking).

        Args:
            client: httpx async client to use
            url: article URL to fetch
            user_agent: User-Agent header value
            max_chars: max characters to return (truncates beyond this)
        """
        if not url:
            return ""

        try:
            headers = {}
            if user_agent:
                headers["User-Agent"] = user_agent

            response = await client.get(
                url,
                follow_redirects=True,
                headers=headers,
                timeout=15.0,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                return ""

            soup = BeautifulSoup(response.text, "lxml")

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                             "iframe", "noscript", "form", "button", "svg"]):
                tag.decompose()

            # Try <article> first, then <main>, then <body>
            main = soup.find("article") or soup.find("main") or soup.find("body")
            if not main:
                return ""

            text = main.get_text(separator="\n", strip=True)
            # Collapse excessive blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)

            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Content truncated]"

            return text

        except Exception as e:
            logger.debug(f"Failed to fetch content from {url}: {e}")
            return ""

    @staticmethod
    async def fetch_url_content_playwright(
        browser,
        url: str,
        timeout_ms: int = 30000,
        max_chars: int = 15000,
    ) -> str:
        """
        Fetch and extract readable text from a URL using Playwright (JS rendering).

        Creates a new browser page, navigates to the URL, waits for network idle,
        then extracts text content via JS. Returns empty string on any failure.

        Args:
            browser: Playwright Browser instance
            url: article URL to fetch
            timeout_ms: page load timeout in milliseconds
            max_chars: max characters to return
        """
        if not url:
            return ""

        # Fail fast if Chromium has died — avoids 80 simultaneous page opens
        # all hitting "Target page, context or browser has been closed".
        if not browser.is_connected():
            return ""

        page = None
        try:
            page = await browser.new_page()
            # domcontentloaded > networkidle on Lambda: most modern sites
            # never reach networkidle (analytics, ads, polling) and the page
            # just hangs until timeout, holding Chromium resources hostage.
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            content = await page.evaluate("""() => {
                const article = document.querySelector('article')
                    || document.querySelector('main')
                    || document.body;
                if (!article) return '';
                const remove = 'script,style,nav,header,footer,aside,iframe,noscript,form,button,svg';
                article.querySelectorAll(remove).forEach(el => el.remove());
                return article.innerText;
            }""")

            if not content:
                return ""

            content = re.sub(r"\n{3,}", "\n\n", content)
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[Content truncated]"
            return content

        except Exception as e:
            logger.warning(f"Playwright failed for {url}: {e}")
            return ""
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass

    @staticmethod
    async def send_discord_webhook(source_name: str, failed_articles: list[dict]) -> None:
        """
        Send failed article info to Discord webhook.

        Args:
            source_name: crawler name (e.g. "HackerNews", "Reddit")
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
