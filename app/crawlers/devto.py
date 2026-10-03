"""Dev.to crawler using REST API"""

from typing import List
from datetime import datetime
import asyncio
import logging
import httpx
from app.crawlers.base import BaseCrawler, RawArticle
from app.crawlers.content import clean_devto_markdown, html_fragment_to_markdown
from app.config.settings import settings

logger = logging.getLogger(__name__)


class DevToCrawler(BaseCrawler):
    """Crawler for Dev.to articles using their public REST API"""

    BASE_URL = "https://dev.to/api/articles"

    async def crawl(self) -> List[RawArticle]:
        """
        Fetch trending articles from Dev.to

        Returns:
            List of RawArticle objects
        """
        self.log_start()
        articles = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await self._retryable_http_request(
                    "GET",
                    self.BASE_URL,
                    client=client,
                    params={"per_page": 50, "top": 7},
                    headers={"User-Agent": self.user_agent},
                )
                data = response.json()

                # Filter on listing metadata first so the per-article detail
                # request is only made for articles we would actually keep.
                candidates = []
                for item in data:
                    try:
                        article = self._parse_listing(item)
                        if not self.should_skip(article):
                            candidates.append(article)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse article: {e}")

                for article in self.drop_known(candidates):
                    article_id = article.raw_data.get("id")
                    body = await self._fetch_full_body(client, article_id) if article_id else ""
                    # Fall back to the listing description if the detail fetch failed
                    article.content = body or article.raw_data.get("description", "")
                    articles.append(article)

                await asyncio.sleep(self.delay)

        except httpx.HTTPError as e:
            self.log_error(e)
        except Exception as e:
            self.log_error(e)

        self.log_end(len(articles))
        return articles

    async def _fetch_full_body(self, client: httpx.AsyncClient, article_id: int) -> str:
        """Fetch full article body from the Dev.to detail API as clean markdown."""
        try:
            response = await self._retryable_http_request(
                "GET",
                f"{self.BASE_URL}/{article_id}",
                client=client,
                headers={"User-Agent": self.user_agent},
                timeout=15.0,
                max_retries=2,
            )
            detail = response.json()
            if detail.get("body_markdown"):
                return clean_devto_markdown(detail["body_markdown"])
            return html_fragment_to_markdown(detail.get("body_html") or "")
        except Exception as e:
            logger.debug(f"Failed to fetch full body for Dev.to article {article_id}: {e}")
            return ""

    def _parse_listing(self, item: dict) -> RawArticle:
        """Parse a Dev.to listing item into RawArticle (body is fetched separately)"""
        published_at = datetime.fromisoformat(
            item["published_at"].replace("Z", "+00:00")
        )

        return RawArticle(
            title_en=item["title"],
            url=item["url"],
            source="devto",
            published_at=published_at,
            tags=item.get("tag_list", []),
            content="",
            upvotes=item.get("positive_reactions_count", 0),
            comments=item.get("comments_count", 0),
            read_time=f"{item.get('reading_time_minutes', 0)} min read",
            raw_data=item
        )

    def should_skip(self, article: RawArticle) -> bool:
        """
        Skip articles with low engagement

        Args:
            article: RawArticle to check

        Returns:
            True if article should be skipped
        """
        min_reactions = settings.MIN_REACTIONS_DEVTO
        return (article.upvotes or 0) < min_reactions
