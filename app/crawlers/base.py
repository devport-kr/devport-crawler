"""Base crawler class with common functionality"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import re
import httpx
from bs4 import BeautifulSoup
from app.config.settings import settings

logger = logging.getLogger(__name__)

MIN_CONTENT_LENGTH = 500


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
    def load_existing_urls() -> set[str]:
        """Load existing article URLs from the database for dedup before expensive retries."""
        try:
            from app.config.database import SessionLocal
            from app.models.article import Article
            db = SessionLocal()
            try:
                urls = {row.url for row in db.query(Article.url).all()}
                logger.info(f"Loaded {len(urls)} existing URLs for pre-dedup")
                return urls
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to load existing URLs (skipping pre-dedup): {e}")
            return set()

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
    async def fetch_url_content_playwright(url: str, max_chars: int = 15000) -> str:
        """
        Retry fetching content using Playwright headless browser.

        Handles JS-rendered pages that httpx can't fetch.
        Returns extracted text or empty string on failure.
        """
        if not url:
            return ""

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=settings.PLAYWRIGHT_HEADLESS)
                try:
                    page = await browser.new_page(
                        user_agent=settings.USER_AGENT,
                    )
                    await page.goto(url, wait_until="domcontentloaded", timeout=settings.PLAYWRIGHT_TIMEOUT)
                    # Give JS a moment to render content
                    await page.wait_for_timeout(2000)

                    html = await page.content()
                finally:
                    await browser.close()

            soup = BeautifulSoup(html, "lxml")

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                             "iframe", "noscript", "form", "button", "svg"]):
                tag.decompose()

            main = soup.find("article") or soup.find("main") or soup.find("body")
            if not main:
                return ""

            text = main.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)

            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Content truncated]"

            return text

        except Exception as e:
            logger.debug(f"Playwright fetch failed for {url}: {e}")
            return ""

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
            f"{len(failed_articles)} articles failed after Playwright retry\n"
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
