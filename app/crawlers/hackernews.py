"""Hacker News crawler using official HN API with Playwright fallback"""

import asyncio
import logging
from collections import Counter
from typing import List, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse
import httpx

from app.crawlers.base import BaseCrawler, RawArticle
from app.crawlers.content import canonicalize_url, html_fragment_to_markdown
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
    - Phase 2: Drop low-engagement/old stories and URLs that are already saved
    - Phase 3: Fetch content for remaining stories in parallel: plain HTTP +
               main-content extraction, Playwright only for pages that fail
               the quality gate (SPAs, bot walls)
    """

    BASE_URL = "https://hacker-news.firebaseio.com/v0"
    HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"

    def __init__(self, known_url_filter=None):
        super().__init__(known_url_filter=known_url_filter)
        self.min_score = getattr(settings, 'MIN_SCORE_HACKERNEWS', 50)
        self.max_age_days = getattr(settings, 'MAX_AGE_DAYS_HACKERNEWS', 7)
        self.max_stories = getattr(settings, 'MAX_STORIES_HACKERNEWS', 100)

    async def crawl(self) -> List[RawArticle]:
        """Fetch top stories from HN with parallel content fetching and Playwright fallback."""
        logger.info(f"Starting HN crawl (min_score={self.min_score}, max_age={self.max_age_days}d)")

        try:
            meta_sem = asyncio.Semaphore(20)

            async with httpx.AsyncClient(timeout=30.0) as client, \
                    httpx.AsyncClient(timeout=20.0, follow_redirects=True) as content_client:
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

                # Phase 3: Filter BEFORE expensive content fetching
                stories_to_fetch = []
                for result in meta_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Error fetching story metadata: {result}")
                        continue
                    if result is not None and not self.should_skip(result):
                        stories_to_fetch.append(result)

                stories_to_fetch = self.drop_known(stories_to_fetch)
                logger.info(
                    f"After filtering: {len(stories_to_fetch)}/{len(story_ids)} stories "
                    f"to fetch content for"
                )

                # Phase 4: Fetch content — HTTP + extraction, Playwright fallback
                async def fetch_content(article: RawArticle) -> RawArticle:
                    original_url = article.raw_data.get("original_url", "")

                    # Ask/Show HN without external URL already have content from story text
                    if not original_url or "news.ycombinator.com" in original_url:
                        return article

                    fetched = await self.fetch_article_content(content_client, original_url)
                    article.content = fetched.markdown
                    article.raw_data["content_method"] = fetched.method
                    article.raw_data["content_issue"] = fetched.issue
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

                methods = Counter(a.raw_data.get("content_method", "hn_text") for a in articles)
                issues = Counter(a.raw_data.get("content_issue") for a in articles if a.raw_data.get("content_issue"))
                logger.info(
                    f"Successfully crawled {len(articles)} HN stories; "
                    f"content methods={dict(methods)} issues={dict(issues)}"
                )
                return articles

        except Exception as e:
            logger.error(f"Error crawling Hacker News: {e}", exc_info=True)
            return []
        finally:
            await self.close_browser()

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

        original_url = canonicalize_url(story.get("url", ""))
        discussion_url = self.HN_ITEM_URL.format(story_id)

        # For Ask HN / Show HN, use the HN post text (HTML) as initial content
        content = ""
        if not original_url and story.get("text"):
            content = html_fragment_to_markdown(story["text"])

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
