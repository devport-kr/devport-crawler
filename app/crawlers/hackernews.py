"""Hacker News crawler using official HN API"""

import asyncio
import logging
from typing import List
from datetime import datetime, timedelta
import httpx
from bs4 import BeautifulSoup

from app.crawlers.base import BaseCrawler, RawArticle, MIN_CONTENT_LENGTH
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

        # Load existing URLs to skip Playwright retry for already-saved articles
        self._existing_urls = self.load_existing_urls()

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

                # Send Discord webhook for articles that failed content fetch
                # Only report articles that pass should_skip (high-engagement, recent)
                failed_articles = []
                for a in articles:
                    original_url = a.raw_data.get("original_url", "")
                    if original_url and len(a.content or "") < MIN_CONTENT_LENGTH and not self.should_skip(a):
                        failed_articles.append({
                            "title": a.title_en,
                            "url": original_url,
                            "discussion_url": a.raw_data.get("hn_discussion_url"),
                            "upvotes": a.upvotes,
                            "comments": a.comments,
                        })

                if failed_articles:
                    logger.info(f"{len(failed_articles)} HN articles failed content fetch (< {MIN_CONTENT_LENGTH} chars)")
                    await self.send_discord_webhook("HackerNews", failed_articles)

                return articles

        except Exception as e:
            logger.error(f"Error crawling Hacker News: {e}")
            return []

    async def _fetch_story(self, client: httpx.AsyncClient, story_id: int) -> RawArticle:
        """Fetch individual story details and article content"""
        response = await client.get(f"{self.BASE_URL}/item/{story_id}.json")
        response.raise_for_status()
        story = response.json()

        if not story:
            raise ValueError(f"Story {story_id} not found")

        original_url = story.get("url", "")
        discussion_url = self.HN_ITEM_URL.format(story_id)

        # Fetch the actual article content from the linked URL
        content = ""
        used_playwright = False
        if original_url and "news.ycombinator.com" not in original_url:
            content = await self.fetch_url_content(client, original_url, self.user_agent)

            # Retry with Playwright if content is too short (skip if URL already in DB)
            if len(content) < MIN_CONTENT_LENGTH and original_url not in self._existing_urls:
                logger.debug(f"httpx got {len(content)} chars for {original_url}, retrying with Playwright")
                pw_content = await self.fetch_url_content_playwright(original_url)
                if len(pw_content) >= MIN_CONTENT_LENGTH:
                    content = pw_content
                    used_playwright = True

        # For Ask HN / Show HN without external URL, use the HN post text if available
        if not content and story.get("text"):
            soup = BeautifulSoup(story["text"], "lxml")
            content = soup.get_text(separator="\n", strip=True)

        article = RawArticle(
            title_en=story.get("title", ""),
            url=original_url or discussion_url,
            source="hackernews",
            published_at=datetime.fromtimestamp(story.get("time", 0)),
            external_id=str(story_id),
            content=content,
            upvotes=story.get("score", 0),
            comments=story.get("descendants", 0),
            language="en",
            raw_data={
                "hn_id": story_id,
                "author": story.get("by", "unknown"),
                "original_url": original_url,
                "hn_discussion_url": discussion_url,
                "story_type": story.get("type", "story"),
                "item_type": "DISCUSSION",
                "used_playwright": used_playwright,
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
