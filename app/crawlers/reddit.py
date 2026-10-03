"""Reddit crawler using public JSON endpoints"""

from collections import Counter
from typing import List
from datetime import datetime, timezone
import asyncio
import httpx
import re
from urllib.parse import urlparse
from app.crawlers.base import BaseCrawler, RawArticle
from app.crawlers.content import canonicalize_url, normalize_markdown
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

    # Subreddits are fetched as multireddits (r/a+b+c) in groups of this size
    SUBREDDITS_PER_REQUEST = 10
    MAX_PAGES_PER_GROUP = 3

    BASE_URL = "https://www.reddit.com/r/{subreddit}/top.json"
    OAUTH_URL = "https://oauth.reddit.com/r/{subreddit}/top.json"
    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

    async def crawl(self) -> List[RawArticle]:
        """
        Fetch top posts from selected subreddits.

        Performance strategy (mirrors HN crawler):
        - Phase 1: Fetch listings for groups of subreddits (r/a+b+c), a handful
                   of requests instead of one per subreddit — Reddit throttles
                   unauthenticated clients to ~10 requests/minute
        - Phase 2: Parse metadata, filter, drop already-saved URLs (no content fetch yet)
        - Phase 3: Fetch external link content in parallel: plain HTTP + extraction,
                   Playwright only for pages that fail the quality gate
        """
        self.log_start()

        try:
            token = await self._get_access_token()
            base_headers = {"User-Agent": settings.REDDIT_USER_AGENT}
            if token:
                base_headers["Authorization"] = f"bearer {token}"
            else:
                self.logger.warning(
                    "REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET not set — using unauthenticated "
                    "Reddit access, which Reddit blocks from most cloud IPs"
                )

            base_url = self.OAUTH_URL if token else self.BASE_URL

            # The Reddit client carries the OAuth token; external article pages are
            # fetched with a separate client so the token never leaves reddit.com.
            async with httpx.AsyncClient(timeout=30.0, headers=base_headers) as client, \
                    httpx.AsyncClient(timeout=20.0, follow_redirects=True) as content_client:
                # Phase 1: Fetch listings per subreddit group, sequentially
                failed_groups: List[str] = []  # HTTP status (or error) of groups that got nothing

                async def fetch_group(subreddits: List[str]) -> List[dict]:
                    posts: List[dict] = []
                    after = None
                    for page in range(self.MAX_PAGES_PER_GROUP):
                        params = {"limit": 100, "t": "day", "raw_json": 1}
                        if after:
                            params["after"] = after
                        try:
                            response = await self._retryable_http_request(
                                "GET",
                                base_url.format(subreddit="+".join(subreddits)),
                                client=client,
                                params=params,
                            )
                        except Exception as e:
                            status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else None
                            if page == 0:
                                failed_groups.append(str(status or type(e).__name__))
                            self.logger.warning(f"Failed to fetch r/{'+'.join(subreddits)}: {e}")
                            break

                        listing = response.json().get("data", {})
                        children = listing.get("children", [])
                        for child in children:
                            post = child.get("data", {})
                            if not post.get("stickied") and not post.get("over_18"):
                                post["__subreddit__"] = post.get("subreddit") or subreddits[0]
                                posts.append(post)

                        # Results are sorted by score: stop once a page dips below the
                        # upvote threshold (later pages can only be lower).
                        after = listing.get("after")
                        lowest = min((c.get("data", {}).get("score", 0) for c in children), default=0)
                        if not after or lowest < settings.MIN_UPVOTES_REDDIT:
                            break
                    return posts

                groups = [
                    self.SUBREDDITS[i:i + self.SUBREDDITS_PER_REQUEST]
                    for i in range(0, len(self.SUBREDDITS), self.SUBREDDITS_PER_REQUEST)
                ]
                self.logger.info(
                    f"Fetching listings from {len(self.SUBREDDITS)} subreddits in {len(groups)} requests..."
                )
                sub_results = []
                for group in groups:
                    sub_results.append(await fetch_group(group))

                if len(failed_groups) == len(groups):
                    self.logger.error(
                        f"Every Reddit listing request failed ({sorted(set(failed_groups))}). "
                        "Reddit blocks unauthenticated API access from most cloud IPs — create a "
                        "Reddit 'script' app and set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET."
                    )

                # Phase 2: Parse metadata, filter, deduplicate (no content fetch)
                seen_urls = set()
                articles_needing_content: List[RawArticle] = []
                articles: List[RawArticle] = []

                for result in sub_results:
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

                articles = self.drop_known(articles)
                articles_needing_content = self.drop_known(articles_needing_content)
                self.logger.info(
                    f"After filtering: {len(articles)} with content, "
                    f"{len(articles_needing_content)} need external content fetch"
                )

                # Phase 3: Fetch external content in parallel
                async def fetch_content(article: RawArticle) -> RawArticle:
                    fetched = await self.fetch_article_content(content_client, article.url)
                    content = fetched.markdown
                    article.content = content
                    article.raw_data["content_method"] = fetched.method
                    article.raw_data["content_issue"] = fetched.issue
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

                fetched = [a for a in articles if "content_method" in a.raw_data]
                self.logger.info(
                    f"External content: methods={dict(Counter(a.raw_data['content_method'] for a in fetched))} "
                    f"issues={dict(Counter(a.raw_data['content_issue'] for a in fetched if a.raw_data['content_issue']))}"
                )

        except Exception as e:
            self.log_error(e)
            articles = []
        finally:
            await self.close_browser()

        self.log_end(len(articles))
        return articles

    def _parse_post_metadata(self, post: dict, subreddit: str) -> RawArticle:
        """Parse Reddit post JSON into RawArticle without fetching external content."""
        created_ts = post.get("created_utc")
        published_at = datetime.fromtimestamp(created_ts, tz=timezone.utc) if created_ts else datetime.utcnow()

        content = normalize_markdown(post.get("selftext") or "")

        url = post.get("url_overridden_by_dest") or post.get("url")
        if not url:
            url = f"https://www.reddit.com{post.get('permalink', '')}"
        url = canonicalize_url(url)

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

        image_ext = re.search(r"\.(png|jpe?g|gif|webp)$", url)
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
            headers = {"User-Agent": settings.REDDIT_USER_AGENT}

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(self.TOKEN_URL, data=data, auth=auth, headers=headers)
                resp.raise_for_status()
                token = resp.json().get("access_token")
                return token
        except Exception as e:
            self.logger.warning(f"Failed to fetch Reddit access token, using public API: {e}")
            return None
