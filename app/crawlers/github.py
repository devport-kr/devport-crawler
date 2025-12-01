"""GitHub trending crawler using GitHub API"""

from typing import List
from datetime import datetime, timedelta
import asyncio
import httpx
from app.crawlers.base import BaseCrawler, RawArticle
from app.config.settings import settings


class GitHubCrawler(BaseCrawler):
    """Crawler for GitHub trending repositories using REST API v3"""

    BASE_URL = "https://api.github.com/search/repositories"

    async def crawl(self) -> List[RawArticle]:
        """
        Fetch trending repositories from GitHub

        Returns:
            List of RawArticle objects
        """
        self.log_start()
        articles = []
        seen_repos = set()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"User-Agent": self.user_agent}

                # Add authentication if token is available
                if settings.GITHUB_TOKEN:
                    headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"

                # Search 1: Repos created in last 7 days, sorted by stars
                created_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
                params = {
                    "q": f"created:>{created_date}",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 50
                }

                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()

                for item in data.get("items", []):
                    try:
                        if item["full_name"] not in seen_repos:
                            article = self._parse_repo(item)
                            if not self.should_skip(article):
                                articles.append(article)
                                seen_repos.add(item["full_name"])
                    except Exception as e:
                        self.logger.warning(f"Failed to parse repo: {e}")
                        continue

                await asyncio.sleep(self.delay)

                # Search 2: Recently updated repos with high stars
                updated_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
                params = {
                    "q": f"stars:>100 pushed:>{updated_date}",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 30
                }

                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()

                for item in data.get("items", []):
                    try:
                        if item["full_name"] not in seen_repos:
                            article = self._parse_repo(item)
                            if not self.should_skip(article):
                                articles.append(article)
                                seen_repos.add(item["full_name"])
                    except Exception as e:
                        self.logger.warning(f"Failed to parse repo: {e}")
                        continue

        except httpx.HTTPError as e:
            self.log_error(e)
        except Exception as e:
            self.log_error(e)

        self.log_end(len(articles))
        return articles

    def _parse_repo(self, item: dict) -> RawArticle:
        """Parse GitHub API response into RawArticle"""
        created_at = datetime.fromisoformat(
            item["created_at"].replace("Z", "+00:00")
        )

        # Use topics as tags, fallback to language
        tags = item.get("topics", [])
        if item.get("language"):
            tags.append(item["language"].lower())

        return RawArticle(
            title_en=item["full_name"],
            url=item["html_url"],
            source="github",
            published_at=created_at,
            tags=tags,
            content=item.get("description", ""),
            stars=item.get("stargazers_count", 0),
            language=item.get("language"),
            raw_data=item
        )

    def should_skip(self, article: RawArticle) -> bool:
        """
        Skip repositories with low stars

        Args:
            article: RawArticle to check

        Returns:
            True if article should be skipped
        """
        min_stars = settings.MIN_STARS_GITHUB
        return (article.stars or 0) < min_stars
