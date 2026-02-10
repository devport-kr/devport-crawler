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
