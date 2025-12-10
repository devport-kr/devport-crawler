"""Medium crawler using RSS feeds"""

from typing import List
from datetime import datetime
import asyncio
import httpx
import feedparser
from app.crawlers.base import BaseCrawler, RawArticle
from app.config.settings import settings


class MediumCrawler(BaseCrawler):
    """Crawler for Medium articles using RSS feeds"""

    # Medium RSS feed URL pattern
    RSS_URL_PATTERN = "https://medium.com/tag/{tag}/feed"

    # Relevant tags for developer content
    TAGS = [
        "programming",
        "javascript",
        "python",
        "devops",
        "cloud-computing",
        "software-engineering",
        "web-development",
        "machine-learning",
        "data-science",
        "backend"
    ]

    async def crawl(self) -> List[RawArticle]:
        """
        Fetch articles from Medium RSS feeds

        Returns:
            List of RawArticle objects
        """
        self.log_start()
        articles = []
        seen_urls = set()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for tag in self.TAGS:
                    try:
                        url = self.RSS_URL_PATTERN.format(tag=tag)
                        response = await client.get(
                            url,
                            headers={"User-Agent": self.user_agent}
                        )
                        response.raise_for_status()

                        # Parse RSS feed
                        feed = feedparser.parse(response.text)

                        for entry in feed.entries[:10]:  # Limit per tag
                            try:
                                # Skip duplicates across tags
                                if entry.link in seen_urls:
                                    continue

                                article = self._parse_entry(entry, tag)
                                if not self.should_skip(article):
                                    articles.append(article)
                                    seen_urls.add(entry.link)
                            except Exception as e:
                                self.logger.warning(f"Failed to parse entry: {e}")
                                continue

                        # Add delay between tags to be polite
                        await asyncio.sleep(self.delay)

                    except Exception as e:
                        self.logger.warning(f"Failed to fetch tag {tag}: {e}")
                        continue

        except Exception as e:
            self.log_error(e)

        self.log_end(len(articles))
        return articles

    def _parse_entry(self, entry, tag: str) -> RawArticle:
        """Parse RSS entry into RawArticle"""
        # Parse published date
        published_at = datetime(*entry.published_parsed[:6])

        # Extract categories/tags
        tags = [tag]
        if hasattr(entry, 'tags'):
            tags.extend([t.term for t in entry.tags])

        # Get content/summary
        content = ""
        if hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'content'):
            content = entry.content[0].value

        # Estimate read time from content length (rough approximation)
        words = len(content.split())
        read_time_minutes = max(1, words // 200)  # Assume 200 words/min

        return RawArticle(
            title_en=entry.title,
            url=entry.link,
            source="medium",
            published_at=published_at,
            tags=tags,
            content=content,
            read_time=f"{read_time_minutes} min read",
            raw_data=dict(entry)
        )

    def should_skip(self, article: RawArticle) -> bool:
        """
        Skip articles based on criteria

        Args:
            article: RawArticle to check

        Returns:
            True if article should be skipped
        """
        # Skip if too old (more than 30 days)
        days_old = (datetime.utcnow() - article.published_at).days
        if days_old > 30:
            return True

        # Check for paywalled content (basic check)
        if "member-only" in article.url.lower():
            return True

        return False
