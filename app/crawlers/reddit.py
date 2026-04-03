"""Reddit crawler using public JSON endpoints"""

from typing import List
from datetime import datetime, timezone
import asyncio
import httpx
import re
from urllib.parse import urlparse
from app.crawlers.base import BaseCrawler, RawArticle
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
        Fetch top posts from selected subreddits.

        Performance strategy (mirrors HN crawler):
        - Phase 1: Fetch all subreddit listings in parallel
        - Phase 2: Parse metadata, filter, deduplicate (no content fetch yet)
        - Phase 3: Fetch external link content in parallel via Playwright
        """
        self.log_start()

        browser = None
        pw = None
        try:
            browser, pw = await self._launch_browser()

            token = await self._get_access_token()
            base_headers = {"User-Agent": self.user_agent}
            if token:
                base_headers["Authorization"] = f"bearer {token}"

            base_url = self.OAUTH_URL if token else self.BASE_URL
            sub_sem = asyncio.Semaphore(10)

            async with httpx.AsyncClient(timeout=30.0, headers=base_headers) as client:
                # Phase 1: Fetch all subreddit listings in parallel
                async def fetch_subreddit(subreddit: str) -> List[dict]:
                    async with sub_sem:
                        try:
                            response = await self._retryable_http_request(
                                "GET",
                                base_url.format(subreddit=subreddit),
                                client=client,
                                params={"limit": 50, "t": "day", "raw_json": 1},
                            )
                            data = response.json()
                            children = data.get("data", {}).get("children", [])
                            posts = []
                            for child in children:
                                post = child.get("data", {})
                                if not post.get("stickied") and not post.get("over_18"):
                                    post["__subreddit__"] = subreddit
                                    posts.append(post)
                            return posts
                        except Exception as e:
                            self.logger.warning(f"Failed to fetch r/{subreddit}: {e}")
                            return []

                self.logger.info(f"Fetching listings from {len(self.SUBREDDITS)} subreddits...")
                sub_results = await asyncio.gather(
                    *[fetch_subreddit(s) for s in self.SUBREDDITS],
                    return_exceptions=True,
                )

                # Phase 2: Parse metadata, filter, deduplicate (no content fetch)
                seen_urls = set()
                articles_needing_content: List[RawArticle] = []
                articles: List[RawArticle] = []

                for result in sub_results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"Subreddit fetch error: {result}")
                        continue
                    for post in result:
                        try:
                            article = self._parse_post_metadata(post, post["__subreddit__"])
                            if article.url in seen_urls:
                                continue
                            if self.should_skip(article):
                                continue
                            seen_urls.add(article.url)

                            # Check if this article needs external content fetching
                            if not article.content and not post.get("is_self") and \
                               not self._extract_domain(article.url).endswith("reddit.com"):
                                articles_needing_content.append(article)
                            else:
                                articles.append(article)
                        except Exception as e:
                            self.logger.warning(f"Failed to parse post: {e}")

                self.logger.info(
                    f"After filtering: {len(articles)} with content, "
                    f"{len(articles_needing_content)} need Playwright fetch"
                )

                # Phase 3: Fetch external content in parallel — Playwright first, httpx fallback
                pw_sem = asyncio.Semaphore(settings.PLAYWRIGHT_CONCURRENCY)
                http_sem = asyncio.Semaphore(settings.CONTENT_FETCH_CONCURRENCY)

                async def fetch_content(article: RawArticle) -> RawArticle:
                    content = ""
                    if browser is not None:
                        async with pw_sem:
                            content = await self.fetch_url_content_playwright(
                                browser, article.url,
                                timeout_ms=settings.PLAYWRIGHT_TIMEOUT_MS,
                            )
                    if not content:
                        async with http_sem:
                            content = await self.fetch_url_content(
                                client, article.url, self.user_agent,
                            )
                    article.content = content
                    # Update read time now that we have content
                    words = len(content.split()) if content else 0
                    if words:
                        article.read_time = f"{max(1, words // 200)} min read"
                    return article

                content_results = await asyncio.gather(
                    *[fetch_content(a) for a in articles_needing_content],
                    return_exceptions=True,
                )

                for result in content_results:
                    if isinstance(result, RawArticle):
                        articles.append(result)
                    elif isinstance(result, Exception):
                        self.logger.warning(f"Content fetch error: {result}")

        except Exception as e:
            self.log_error(e)
            articles = []
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

        self.log_end(len(articles))
        return articles

    def _parse_post_metadata(self, post: dict, subreddit: str) -> RawArticle:
        """Parse Reddit post JSON into RawArticle without fetching external content."""
        created_ts = post.get("created_utc")
        published_at = datetime.fromtimestamp(created_ts, tz=timezone.utc) if created_ts else datetime.utcnow()

        content = post.get("selftext") or ""

        url = post.get("url_overridden_by_dest") or post.get("url")
        if not url:
            url = f"https://www.reddit.com{post.get('permalink', '')}"

        domain = self._extract_domain(url)
        is_self = post.get("is_self")
        source = "reddit" if is_self or domain.endswith("reddit.com") else domain

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
            raw_data=post,
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
