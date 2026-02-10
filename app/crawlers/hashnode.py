"""Hashnode crawler using GraphQL API"""

from typing import List
from datetime import datetime
import asyncio
import httpx
from app.crawlers.base import BaseCrawler, RawArticle
from app.config.settings import settings


class HashnodeCrawler(BaseCrawler):
    """Crawler for Hashnode articles using their GraphQL API"""

    GRAPHQL_URL = "https://gql.hashnode.com/"

    QUERY = """
    query {
      feed(first: 50, filter: {type: FEATURED}) {
        edges {
          node {
            id
            title
            url
            publishedAt
            reactionCount
            responseCount
            readTimeInMinutes
            tags {
              name
            }
            brief
            content {
              markdown
            }
          }
        }
      }
    }
    """

    async def crawl(self) -> List[RawArticle]:
        """
        Fetch featured articles from Hashnode

        Returns:
            List of RawArticle objects
        """
        self.log_start()
        articles = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.GRAPHQL_URL,
                    json={"query": self.QUERY},
                    headers={
                        "User-Agent": self.user_agent,
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                data = response.json()

                edges = data.get("data", {}).get("feed", {}).get("edges", [])

                for edge in edges:
                    try:
                        node = edge.get("node", {})
                        article = self._parse_article(node)
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

    def _parse_article(self, node: dict) -> RawArticle:
        """Parse Hashnode GraphQL response into RawArticle"""
        published_at = datetime.fromisoformat(
            node["publishedAt"].replace("Z", "+00:00")
        )

        tags = [tag["name"] for tag in node.get("tags", [])]

        read_time = node.get("readTimeInMinutes", 0)

        # Prefer full markdown content, fall back to brief
        content_obj = node.get("content") or {}
        content = content_obj.get("markdown") or node.get("brief", "")

        return RawArticle(
            title_en=node["title"],
            url=node["url"],
            source="hashnode",
            published_at=published_at,
            tags=tags,
            content=content,
            upvotes=node.get("reactionCount", 0),
            comments=node.get("responseCount", 0),
            read_time=f"{read_time} min read" if read_time else None,
            raw_data=node
        )

    def should_skip(self, article: RawArticle) -> bool:
        """
        Skip articles with low engagement

        Args:
            article: RawArticle to check

        Returns:
            True if article should be skipped
        """
        min_reactions = settings.MIN_REACTIONS_HASHNODE
        return (article.upvotes or 0) < min_reactions
