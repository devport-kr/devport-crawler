"""Hacker News crawler using official HN API"""

import asyncio
import logging
from typing import List
from datetime import datetime, timedelta
import httpx

from app.crawlers.base import BaseCrawler, RawArticle
from app.config.settings import settings

logger = logging.getLogger(__name__)


class HackerNewsCrawler(BaseCrawler):
    """
    Crawls Hacker News using the official Firebase API

    API Docs: https://github.com/HackerNews/API

    Uses the original article URL as the primary URL when available.
    Falls back to the HN discussion URL for Ask/Show HN and other items
    without an external link. The discussion URL is always stored in raw_data.
    """

    BASE_URL = "https://hacker-news.firebaseio.com/v0"
    HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"

    def __init__(self):
        super().__init__()
        self.min_score = getattr(settings, 'MIN_SCORE_HACKERNEWS', 50)
        self.max_age_days = getattr(settings, 'MAX_AGE_DAYS_HACKERNEWS', 7)
        self.max_stories = getattr(settings, 'MAX_STORIES_HACKERNEWS', 100)

    async def crawl(self) -> List[RawArticle]:
        """Fetch top stories from Hacker News"""
        logger.info(f"Starting Hacker News crawl (min_score={self.min_score}, max_age={self.max_age_days}d)")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get top story IDs
                response = await client.get(f"{self.BASE_URL}/topstories.json")
                response.raise_for_status()
                story_ids = response.json()[:self.max_stories]  # Limit to top N

                logger.info(f"Fetched {len(story_ids)} top story IDs")

                # Fetch story details in batches
                articles = []
                batch_size = 10

                for i in range(0, len(story_ids), batch_size):
                    batch_ids = story_ids[i:i + batch_size]
                    batch_tasks = [self._fetch_story(client, story_id) for story_id in batch_ids]
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                    for result in batch_results:
                        if isinstance(result, RawArticle):
                            articles.append(result)
                        elif isinstance(result, Exception):
                            logger.warning(f"Error fetching story: {result}")

                    # Rate limiting
                    await asyncio.sleep(0.5)

                logger.info(f"Successfully crawled {len(articles)} HN stories")
                return articles

        except Exception as e:
            logger.error(f"Error crawling Hacker News: {e}")
            return []

    async def _fetch_story(self, client: httpx.AsyncClient, story_id: int) -> RawArticle:
        """Fetch individual story details"""
        response = await client.get(f"{self.BASE_URL}/item/{story_id}.json")
        response.raise_for_status()
        story = response.json()

        if not story:
            raise ValueError(f"Story {story_id} not found")

        # Convert to RawArticle
        original_url = story.get("url", "")
        discussion_url = self.HN_ITEM_URL.format(story_id)

        article = RawArticle(
            title_en=story.get("title", ""),
            url=original_url or discussion_url,  # Prefer original source URL
            source="hackernews",
            published_at=datetime.fromtimestamp(story.get("time", 0)),
            external_id=str(story_id),
            upvotes=story.get("score", 0),
            comments=story.get("descendants", 0),  # Total comment count
            language="en",
            raw_data={
                "hn_id": story_id,
                "author": story.get("by", "unknown"),
                "original_url": original_url,
                "hn_discussion_url": discussion_url,
                "story_type": story.get("type", "story"),
                "item_type": "DISCUSSION",  # Store as metadata
            }
        )

        return article

    def should_skip(self, article: RawArticle) -> bool:
        """Filter out low-engagement or old stories"""

        # Skip if no URL (shouldn't happen with our approach)
        if not article.url:
            logger.debug(f"Skipping story without URL: {article.title_en}")
            return True

        # Skip Ask HN, Show HN, Job posts without original URL
        original_url = article.raw_data.get("original_url", "")
        if not original_url:
            # Allow if it's explicitly Ask HN or Show HN (discussion itself is valuable)
            if not (article.title_en.startswith("Ask HN:") or article.title_en.startswith("Show HN:")):
                logger.debug(f"Skipping story without original URL: {article.title_en}")
                return True

        # Skip low engagement
        if article.upvotes < self.min_score:
            logger.debug(f"Skipping low-score story: {article.title_en} ({article.upvotes} points)")
            return True

        # Skip old stories
        age = datetime.utcnow() - article.published_at
        if age > timedelta(days=self.max_age_days):
            logger.debug(f"Skipping old story: {article.title_en} ({age.days} days old)")
            return True

        return False
