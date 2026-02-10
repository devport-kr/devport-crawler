"""Dev.to crawler using REST API"""

from typing import List
from datetime import datetime
import asyncio
import logging
import httpx
from app.crawlers.base import BaseCrawler, RawArticle
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
                response = await client.get(
                    self.BASE_URL,
                    params={
                        "per_page": 50,
                        "top": 7
                    },
                    headers={"User-Agent": self.user_agent}
                )
                response.raise_for_status()
                data = response.json()

                for item in data:
                    try:
                        article = await self._parse_article(client, item)
                        if not self.should_skip(article):
                            articles.append(article)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse article: {e}")
                        continue

                await asyncio.sleep(self.delay)

        except httpx.HTTPError as e:
            self.log_error(e)
        except Exception as e:
            self.log_error(e)

        self.log_end(len(articles))
        return articles

    async def _fetch_full_body(self, client: httpx.AsyncClient, article_id: int) -> str:
        """Fetch full article body markdown from the Dev.to detail API."""
        try:
            response = await client.get(
                f"{self.BASE_URL}/{article_id}",
                headers={"User-Agent": self.user_agent},
                timeout=15.0,
            )
            response.raise_for_status()
            detail = response.json()
            return detail.get("body_markdown") or detail.get("body_html") or ""
        except Exception as e:
            logger.debug(f"Failed to fetch full body for Dev.to article {article_id}: {e}")
            return ""

    async def _parse_article(self, client: httpx.AsyncClient, item: dict) -> RawArticle:
        """Parse Dev.to API response into RawArticle, fetching full body"""
        published_at = datetime.fromisoformat(
            item["published_at"].replace("Z", "+00:00")
        )

        # Fetch full article body from detail endpoint
        article_id = item.get("id")
        content = ""
        if article_id:
            content = await self._fetch_full_body(client, article_id)

        # Fall back to description if detail fetch failed
        if not content:
            content = item.get("description", "")

        return RawArticle(
            title_en=item["title"],
            url=item["url"],
            source="devto",
            published_at=published_at,
            tags=item.get("tag_list", []),
            content=content,
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
