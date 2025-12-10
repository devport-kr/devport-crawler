"""GitHub trending crawler using BeautifulSoup to scrape trending page"""

from typing import List
from datetime import datetime
import asyncio
import httpx
from bs4 import BeautifulSoup
from app.crawlers.base import BaseCrawler, RawArticle
from app.config.settings import settings


class GitHubCrawler(BaseCrawler):
    """Crawler for GitHub trending repositories by scraping trending page"""

    TRENDING_URL = "https://github.com/trending?since=weekly"

    async def crawl(self) -> List[RawArticle]:
        """
        Fetch trending repositories from GitHub trending page

        Returns:
            List of RawArticle objects
        """
        self.log_start()
        articles = []

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                headers = {
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5"
                }

                self.logger.info(f"Fetching {self.TRENDING_URL}")
                response = await client.get(self.TRENDING_URL, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                repo_elements = soup.find_all("article", class_="Box-row")
                self.logger.info(f"Found {len(repo_elements)} trending repositories")

                for repo_element in repo_elements:
                    try:
                        repo_link = repo_element.find("h2").find("a")
                        if not repo_link:
                            continue

                        repo_path = repo_link.get("href", "")
                        repo_full_name = repo_path.strip("/")
                        repo_url = f"https://github.com{repo_path}"

                        desc_element = repo_element.find("p")
                        description = desc_element.get_text(strip=True) if desc_element else ""

                        lang_element = repo_element.find("span", {"itemprop": "programmingLanguage"})
                        language = lang_element.get_text(strip=True) if lang_element else None

                        star_link = repo_element.find("a", href=lambda h: h and "/stargazers" in h)
                        stars_text = star_link.get_text(strip=True) if star_link else "0"
                        stars = self._parse_star_count(stars_text)

                        fork_link = repo_element.find("a", href=lambda h: h and "/forks" in h)
                        forks_text = fork_link.get_text(strip=True) if fork_link else "0"
                        forks = self._parse_star_count(forks_text)

                        stars_this_week_span = repo_element.find("span", class_="d-inline-block float-sm-right")
                        if stars_this_week_span:
                            stars_this_week_text = stars_this_week_span.get_text(strip=True).split()[0]
                            stars_this_week = self._parse_star_count(stars_this_week_text)
                        else:
                            stars_this_week = 0

                        article = RawArticle(
                            title_en=repo_full_name,
                            url=repo_url,
                            source="github",
                            published_at=datetime.utcnow(),  # Trending repos don't have a specific date
                            tags=[language.lower()] if language else [],
                            content=description,
                            stars=stars,
                            language=language,
                            raw_data={
                                "forks": forks,
                                "stars_this_week": stars_this_week
                            }
                        )

                        if not self.should_skip(article):
                            articles.append(article)

                    except Exception as e:
                        self.logger.warning(f"Failed to parse trending repo: {e}")
                        continue

        except Exception as e:
            self.log_error(e)

        self.log_end(len(articles))
        return articles

    def _parse_star_count(self, text: str) -> int:
        """Parse star count from text like '1,234' or '1.2k' to integer"""
        try:
            text = text.replace(",", "").strip()
            if "k" in text.lower():
                return int(float(text.lower().replace("k", "")) * 1000)
            return int(text)
        except:
            return 0

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
