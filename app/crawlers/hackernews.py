"""Hacker News crawler using official HN API with Playwright fallback"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup

from app.crawlers.base import BaseCrawler, RawArticle
from app.config.settings import settings

logger = logging.getLogger(__name__)


class HackerNewsCrawler(BaseCrawler):
    """
    Crawls Hacker News using the official Firebase API.

    API Docs: https://github.com/HackerNews/API

    Uses the original article URL as the primary URL when available.
    Falls back to the HN discussion URL for Ask/Show HN and other items
    without an external link. The discussion URL is always stored in raw_data.

    Performance strategy:
    - Phase 1: Fetch all story metadata in parallel (lightweight API calls)
    - Phase 2: Apply should_skip() filter to drop low-engagement/old stories
    - Phase 3: Fetch content for remaining stories in parallel,
               using httpx first with Playwright JS-rendering fallback
    """

    BASE_URL = "https://hacker-news.firebaseio.com/v0"
    HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"

    def __init__(self):
        super().__init__()
        self.min_score = getattr(settings, 'MIN_SCORE_HACKERNEWS', 50)
        self.max_age_days = getattr(settings, 'MAX_AGE_DAYS_HACKERNEWS', 7)
        self.max_stories = getattr(settings, 'MAX_STORIES_HACKERNEWS', 100)

    async def crawl(self) -> List[RawArticle]:
        """Fetch top stories from HN with parallel content fetching and Playwright fallback."""
        logger.info(f"Starting HN crawl (min_score={self.min_score}, max_age={self.max_age_days}d)")

        browser = None
        pw = None
        try:
            # Launch Playwright browser (shared across all content fetches)
            browser, pw = await self._launch_browser()

            http_sem = asyncio.Semaphore(settings.CONTENT_FETCH_CONCURRENCY)
            pw_sem = asyncio.Semaphore(settings.PLAYWRIGHT_CONCURRENCY)
            meta_sem = asyncio.Semaphore(20)

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Phase 1: Get top story IDs
                response = await self._retryable_http_request(
                    "GET", f"{self.BASE_URL}/topstories.json", client=client,
                )
                story_ids = response.json()[:self.max_stories]
                logger.info(f"Fetched {len(story_ids)} top story IDs")

                # Phase 2: Fetch all story metadata in parallel (no content yet)
                async def fetch_meta(sid: int):
                    async with meta_sem:
                        return await self._fetch_story_metadata(client, sid)

                meta_results = await asyncio.gather(
                    *[fetch_meta(sid) for sid in story_ids],
                    return_exceptions=True,
                )

                # Phase 3: Apply should_skip filter BEFORE expensive content fetching
                stories_to_fetch = []
                for result in meta_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Error fetching story metadata: {result}")
                        continue
                    if result is not None and not self.should_skip(result):
                        stories_to_fetch.append(result)

                logger.info(
                    f"After filtering: {len(stories_to_fetch)}/{len(story_ids)} stories "
                    f"to fetch content for"
                )

                # Phase 4: Fetch content — Playwright first, httpx fallback
                async def fetch_content(article: RawArticle) -> RawArticle:
                    original_url = article.raw_data.get("original_url", "")

                    # Ask/Show HN without external URL already have content from story text
                    if not original_url or "news.ycombinator.com" in original_url:
                        return article

                    content = ""

                    # Primary: Playwright with JS rendering
                    if browser is not None:
                        async with pw_sem:
                            content = await self.fetch_url_content_playwright(
                                browser,
                                original_url,
                                timeout_ms=settings.PLAYWRIGHT_TIMEOUT_MS,
                            )
                        if content:
                            logger.info(f"Playwright got {len(content)} chars from {original_url}")
                            article.raw_data["used_playwright"] = True

                    # Fallback: httpx + BeautifulSoup
                    if not content:
                        async with http_sem:
                            content = await self.fetch_url_content(
                                client, original_url, self.user_agent
                            )
                        if content:
                            logger.info(f"httpx fallback got {len(content)} chars from {original_url}")
                        article.raw_data["used_playwright"] = False

                    article.content = content
                    return article

                content_results = await asyncio.gather(
                    *[fetch_content(a) for a in stories_to_fetch],
                    return_exceptions=True,
                )

                articles = []
                for result in content_results:
                    if isinstance(result, RawArticle):
                        articles.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Error fetching content: {result}")

                logger.info(f"Successfully crawled {len(articles)} HN stories")
                return articles

        except Exception as e:
            logger.error(f"Error crawling Hacker News: {e}", exc_info=True)
            return []
        finally:
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

    async def _fetch_story_metadata(
        self, client: httpx.AsyncClient, story_id: int
    ) -> Optional[RawArticle]:
        """Fetch story metadata from HN API without fetching external content."""
        response = await self._retryable_http_request(
            "GET",
            f"{self.BASE_URL}/item/{story_id}.json",
            client=client,
            max_retries=2,
        )
        story = response.json()

        if not story:
            return None

        original_url = story.get("url", "")
        discussion_url = self.HN_ITEM_URL.format(story_id)

        # For Ask HN / Show HN, use the HN post text as initial content
        content = ""
        if not original_url and story.get("text"):
            soup = BeautifulSoup(story["text"], "lxml")
            content = soup.get_text(separator="\n", strip=True)

        # Extract domain from original URL as source, fallback to "hackernews"
        source = "hackernews"
        if original_url:
            parsed = urlparse(original_url)
            if parsed.netloc:
                source = parsed.netloc

        return RawArticle(
            title_en=story.get("title", ""),
            url=original_url or discussion_url,
            source=source,
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
            }
        )

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
