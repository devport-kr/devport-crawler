"""Reddit crawler using public JSON endpoints"""

from typing import List
from datetime import datetime, timezone
import asyncio
import httpx
import re
from urllib.parse import urlparse
from app.crawlers.base import BaseCrawler, RawArticle, MIN_CONTENT_LENGTH
from app.config.settings import settings


class RedditCrawler(BaseCrawler):
    """Crawler for Reddit developer subreddits using the public JSON API"""

    SUBREDDITS = [
    "programming", "technology", "softwareengineering", "computerscience",

    "MachineLearning", "LocalLLaMA", "deeplearning", "ArtificialInteligence", "OpenAI", "ChatGPT",

    "devops", "sre", "kubernetes", "docker", "aws", "googlecloud", "azure", "cloud",
    "linux", "sysadmin", "networking", "homelab", "selfhosted",

    "database", "postgresql", "mysql", "bigdata", "dataengineering", "datascience",

    "netsec", "cybersecurity", "reverseengineering", "malware", "cryptography",

    "blockchain", "ethereum", "ethdev", "solidity", "web3",

    "webdev", "javascript", "reactjs", "nextjs", "vuejs", "typescript", "html", "css",

    "backend", "api", "programminglanguages",
    "java", "spring", "golang", "rust", "cpp", "python", "dotnet",

    "androiddev", "iOSProgramming", "flutter", "reactnative",

    "softwarearchitecture", "systemdesign", "scalability", "distributed",

    "opensource", "tech", "startup"
    ]

    BASE_URL = "https://www.reddit.com/r/{subreddit}/top.json"
    OAUTH_URL = "https://oauth.reddit.com/r/{subreddit}/top.json"
    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

    async def crawl(self) -> List[RawArticle]:
        """
        Fetch top posts from selected subreddits

        Returns:
            List of RawArticle objects
        """
        self.log_start()
        articles: List[RawArticle] = []
        seen_urls = set()

        # Load existing URLs to skip Playwright retry for already-saved articles
        self._existing_urls = self.load_existing_urls()

        try:
            token = await self._get_access_token()
            base_headers = {"User-Agent": self.user_agent}
            if token:
                base_headers["Authorization"] = f"bearer {token}"

            base_url = self.OAUTH_URL if token else self.BASE_URL

            async with httpx.AsyncClient(timeout=30.0, headers=base_headers) as client:
                for subreddit in self.SUBREDDITS:
                    try:
                        response = await client.get(
                            base_url.format(subreddit=subreddit),
                            params={"limit": 50, "t": "day", "raw_json": 1},
                        )
                        response.raise_for_status()
                        data = response.json()
                        children = data.get("data", {}).get("children", [])

                        for child in children:
                            post = child.get("data", {})

                            # Skip stickied or NSFW posts early
                            if post.get("stickied") or post.get("over_18"):
                                continue

                            try:
                                article = await self._parse_post(client, post, subreddit)

                                # Skip duplicates across subreddits
                                if article.url in seen_urls:
                                    continue

                                if not self.should_skip(article):
                                    articles.append(article)
                                    seen_urls.add(article.url)
                            except Exception as e:
                                self.logger.warning(f"Failed to parse post in r/{subreddit}: {e}")
                                continue

                        await asyncio.sleep(self.delay)

                    except httpx.HTTPError as e:
                        self.logger.warning(f"HTTP error for r/{subreddit}: {e}")
                        await asyncio.sleep(self.delay)
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch r/{subreddit}: {e}")
                        await asyncio.sleep(self.delay)
                        continue

        except Exception as e:
            self.log_error(e)

        # Send Discord webhook for link posts that failed content fetch
        # Only report articles that pass should_skip (high-engagement, non-media)
        failed_articles = []
        for a in articles:
            is_link_post = not a.raw_data.get("is_self") and not a.raw_data.get("domain", "").endswith("reddit.com")
            if is_link_post and len(a.content or "") < MIN_CONTENT_LENGTH and not self.should_skip(a):
                permalink = a.raw_data.get("permalink", "")
                failed_articles.append({
                    "title": a.title_en,
                    "url": a.url,
                    "discussion_url": f"https://www.reddit.com{permalink}" if permalink else None,
                    "upvotes": a.upvotes,
                    "comments": a.comments,
                })

        if failed_articles:
            self.logger.info(f"{len(failed_articles)} Reddit articles failed content fetch (< {MIN_CONTENT_LENGTH} chars)")
            await self.send_discord_webhook("Reddit", failed_articles)

        self.log_end(len(articles))
        return articles

    async def _parse_post(self, client: httpx.AsyncClient, post: dict, subreddit: str) -> RawArticle:
        """Parse Reddit post JSON into RawArticle, fetching linked content for link posts"""
        created_ts = post.get("created_utc")
        published_at = datetime.fromtimestamp(created_ts, tz=timezone.utc) if created_ts else datetime.utcnow()

        content = post.get("selftext") or ""

        # Prefer external link destination, fallback to Reddit permalink
        url = post.get("url_overridden_by_dest") or post.get("url")
        if not url:
            url = f"https://www.reddit.com{post.get('permalink', '')}"

        # Determine source: external domain if link post, otherwise reddit
        domain = self._extract_domain(url)
        is_self = post.get("is_self")
        source = "reddit" if is_self or domain.endswith("reddit.com") else domain

        # For link posts without selftext, fetch the linked article content
        used_playwright = False
        if not content and not is_self and not domain.endswith("reddit.com"):
            content = await self.fetch_url_content(client, url, self.user_agent)

            # Retry with Playwright if content is too short (skip if URL already in DB)
            if len(content) < MIN_CONTENT_LENGTH and url not in self._existing_urls:
                self.logger.debug(f"httpx got {len(content)} chars for {url}, retrying with Playwright")
                pw_content = await self.fetch_url_content_playwright(url)
                if len(pw_content) >= MIN_CONTENT_LENGTH:
                    content = pw_content
                    used_playwright = True

        # Rough read time estimation based on content
        words = len(content.split()) if content else 0
        read_time_minutes = max(1, words // 200) if words else None
        read_time = f"{read_time_minutes} min read" if read_time_minutes else None

        tags = [subreddit, source] if source != "reddit" else [subreddit]

        return RawArticle(
            title_en=post.get("title", "Untitled"),
            url=url,
            source=source,
            published_at=published_at,
            tags=tags,
            content=content,
            upvotes=post.get("score") or post.get("ups") or 0,
            comments=post.get("num_comments", 0),
            read_time=read_time,
            raw_data={**post, "used_playwright": used_playwright},
        )

    def should_skip(self, article: RawArticle) -> bool:
        """
        Skip posts with low engagement or NSFW flag

        Args:
            article: RawArticle to check

        Returns:
            True if article should be skipped
        """
        if article.raw_data.get("over_18"):
            return True

        # Skip image-only or media posts without textual content
        raw = article.raw_data
        post_hint = raw.get("post_hint")
        is_gallery = raw.get("is_gallery")
        url = article.url.lower()
        has_text = bool(article.content and article.content.strip())

        image_ext = re.search(r"\\.(png|jpe?g|gif|webp)$", url)
        is_image_host = any(host in url for host in ["i.redd.it", "i.imgur.com"])

        if not has_text and (post_hint in {"image", "rich:video", "hosted:video"} or is_gallery or image_ext or is_image_host):
            return True

        min_upvotes = settings.MIN_UPVOTES_REDDIT
        return (article.upvotes or 0) < min_upvotes

    @staticmethod
    def _extract_domain(url: str) -> str:
        """
        Extract hostname from URL, stripping www and ignoring path/query.
        """
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path  # handles scheme-less URLs
        host = host.split("/")[0].lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    async def _get_access_token(self) -> str | None:
        """
        Get OAuth access token if client credentials are configured.

        Returns:
            Access token string or None on failure/missing config
        """
        client_id = settings.REDDIT_CLIENT_ID
        client_secret = settings.REDDIT_CLIENT_SECRET
        if not client_id or not client_secret:
            return None

        try:
            auth = (client_id, client_secret)
            data = {"grant_type": "client_credentials"}
            headers = {"User-Agent": self.user_agent}

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(self.TOKEN_URL, data=data, auth=auth, headers=headers)
                resp.raise_for_status()
                token = resp.json().get("access_token")
                return token
        except Exception as e:
            self.logger.warning(f"Failed to fetch Reddit access token, using public API: {e}")
            return None
