"""Orchestrator to coordinate crawler execution and data processing"""

from collections import Counter
from typing import List, Dict, Any
from datetime import datetime
import asyncio
import logging
import uuid
from sqlalchemy.orm import Session

from app.crawlers.devto import DevToCrawler
from app.crawlers.hashnode import HashnodeCrawler
# from app.crawlers.medium import MediumCrawler  # Disabled: ~70% of articles are RSS excerpts, not full content
from app.crawlers.reddit import RedditCrawler
from app.crawlers.hackernews import HackerNewsCrawler
from app.crawlers.github import GitHubCrawler
from app.crawlers.llm_rankings import LLMRankingsCrawler
from app.crawlers.llm_media_rankings import LLMMediaRankingsCrawler
from app.crawlers.base import BaseCrawler, RawArticle
from app.crawlers.content import assess_content
from app.services.summarizer import SummarizerService, SummaryResult
from app.services.scorer import ScorerService
from app.services.deduplicator import DeduplicatorService
from app.services.webhook_dispatcher import dispatch_completion_webhook
from app.models.article import Article, ItemType, Category
from app.models.article_tag import ArticleTag
from app.models.git_repo import GitRepo
from app.config.database import SessionLocal
from app.config.settings import settings

logger = logging.getLogger(__name__)


class CrawlerOrchestrator:
    """Orchestrates the crawling, processing, and storage of articles"""

    def __init__(self):
        self.summarizer = SummarizerService()
        self.scorer = ScorerService()

    async def run_all_crawlers(self) -> Dict[str, Any]:
        """
        Run all blog crawlers concurrently.

        Each source crawls and processes articles independently via asyncio.gather.
        DB safety: each _process_and_save_articles call creates its own SessionLocal.

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting all blog crawlers (concurrent)...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "sources": {},
            "total_crawled": 0,
            "total_saved": 0,
            "errors": []
        }

        sources = [
            ("devto", DevToCrawler(known_url_filter=self._known_article_urls)),
            ("hashnode", HashnodeCrawler()),
            # ("medium", MediumCrawler()),  # Disabled: ~70% of articles are RSS excerpts, not full content
            ("reddit", RedditCrawler(known_url_filter=self._known_article_urls)),
            ("hackernews", HackerNewsCrawler(known_url_filter=self._known_article_urls))
        ]

        async def run_source(source_name: str, crawler) -> tuple:
            try:
                logger.info(f"Running {source_name} crawler...")
                articles = await crawler.crawl()
                saved = await self._process_and_save_articles(articles)
                return source_name, {
                    "crawled": len(articles),
                    "saved": saved,
                    "success": True,
                }
            except Exception as e:
                logger.error(f"Error crawling {source_name}: {e}", exc_info=True)
                return source_name, {
                    "crawled": 0,
                    "saved": 0,
                    "success": False,
                    "error": str(e),
                }

        results = await asyncio.gather(
            *[run_source(name, crawler) for name, crawler in sources],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                stats["errors"].append(str(result))
                continue
            source_name, source_stats = result
            stats["sources"][source_name] = source_stats
            stats["total_crawled"] += source_stats["crawled"]
            stats["total_saved"] += source_stats["saved"]
            if not source_stats["success"]:
                stats["errors"].append(f"{source_name}: {source_stats.get('error', 'unknown')}")

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"All crawlers completed. Total saved: {stats['total_saved']}")

        return stats

    async def run_github_crawler(self) -> Dict[str, Any]:
        """
        Run GitHub trending crawler

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting GitHub crawler...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "source": "github",
            "crawled": 0,
            "saved": 0,
            "success": False
        }

        try:
            crawler = GitHubCrawler()
            repos = await crawler.crawl()
            saved = await self._process_and_save_repositories(repos)

            stats["crawled"] = len(repos)
            stats["saved"] = saved
            stats["success"] = True

            # Notify Spring API to invalidate GIT_REPO caches after the
            # snapshot has been committed. Only dispatched on success so a
            # failed crawl doesn't evict caches pointing at good data.
            webhook_result = await dispatch_completion_webhook(
                scope="GIT_REPO",
                job_id=f"github-{stats['started_at']}",
                completed_at=datetime.utcnow().isoformat(),
            )
            if webhook_result is not None:
                stats["webhook"] = webhook_result

        except Exception as e:
            logger.error(f"Error crawling GitHub: {e}", exc_info=True)
            stats["error"] = str(e)

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"GitHub crawler completed. Saved: {stats['saved']}")

        return stats

    async def run_llm_crawler(self) -> Dict[str, Any]:
        """
        Run LLM rankings crawler using Artificial Analysis API

        Fetches LLM model data including pricing, performance, and 18 benchmark scores.
        Also fetches model creators (providers/organizations).
        Upserts data into model_creators and llm_models tables (UPDATE existing, INSERT new).

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting LLM rankings crawler...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "source": "llm_rankings",
            "creators_crawled": 0,
            "models_crawled": 0,
            "creators_saved": 0,
            "models_saved": 0,
            "success": False
        }

        db = SessionLocal()

        try:
            # Create crawler with database session
            crawler = LLMRankingsCrawler(db=db)

            # Fetch creators and models from Artificial Analysis API
            data = await crawler.crawl()
            creators = data.get("creators", [])
            models = data.get("models", [])

            logger.info(f"Fetched {len(creators)} creators and {len(models)} models from API")

            # Save creators and models to database (upsert)
            save_result = await crawler.save_data(data)

            stats["creators_crawled"] = len(creators)
            stats["models_crawled"] = len(models)
            stats["creators_saved"] = save_result.get("creators", 0)
            stats["models_saved"] = save_result.get("models", 0)
            stats["success"] = True

            logger.info(
                f"LLM crawler completed: {stats['creators_saved']}/{len(creators)} creators saved, "
                f"{stats['models_saved']}/{len(models)} models saved"
            )

        except Exception as e:
            logger.error(f"Error crawling LLM rankings: {e}", exc_info=True)
            stats["error"] = str(e)
        finally:
            db.close()

        stats["completed_at"] = datetime.utcnow().isoformat()

        return stats

    async def run_llm_media_crawler(self) -> Dict[str, Any]:
        """
        Run LLM media rankings crawler using Artificial Analysis API

        Fetches media model data for:
        - text-to-image
        - image-editing
        - text-to-speech
        - text-to-video
        - image-to-video

        Upserts model creators and media models with categories where available.
        """
        logger.info("Starting LLM media rankings crawler...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "source": "llm_media_rankings",
            "creators_crawled": 0,
            "creators_saved": 0,
            "media": {},
            "success": False,
        }

        db = SessionLocal()

        try:
            crawler = LLMMediaRankingsCrawler(db=db)
            data = await crawler.crawl()
            creators = data.get("creators", [])
            media = data.get("media", {})

            stats["creators_crawled"] = len(creators)
            stats["media"] = {k: {"crawled": len(v)} for k, v in media.items()}

            save_result = await crawler.save_data(data)
            stats["creators_saved"] = save_result.get("creators", {}).get("saved", 0)

            for media_type, result in save_result.items():
                if media_type == "creators":
                    continue
                stats["media"].setdefault(media_type, {})
                stats["media"][media_type].update({
                    "saved": result.get("saved", 0),
                    "categories_saved": result.get("categories_saved", 0),
                })

            stats["success"] = True

            logger.info(
                "LLM media crawler completed: %s creators saved, media types: %s",
                stats["creators_saved"],
                ", ".join(stats["media"].keys()),
            )

        except Exception as e:
            logger.error(f"Error crawling LLM media rankings: {e}", exc_info=True)
            stats["error"] = str(e)
        finally:
            db.close()

        stats["completed_at"] = datetime.utcnow().isoformat()
        return stats

    async def run_devto_crawler(self) -> Dict[str, Any]:
        """
        Run Dev.to crawler

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting Dev.to crawler...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "source": "devto",
            "crawled": 0,
            "saved": 0,
            "success": False
        }

        try:
            crawler = DevToCrawler(known_url_filter=self._known_article_urls)
            articles = await crawler.crawl()
            saved = await self._process_and_save_articles(articles)

            stats["crawled"] = len(articles)
            stats["saved"] = saved
            stats["success"] = True

        except Exception as e:
            logger.error(f"Error crawling Dev.to: {e}", exc_info=True)
            stats["error"] = str(e)

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Dev.to crawler completed. Saved: {stats['saved']}")

        return stats

    async def run_hashnode_crawler(self) -> Dict[str, Any]:
        """
        Run Hashnode crawler

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting Hashnode crawler...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "source": "hashnode",
            "crawled": 0,
            "saved": 0,
            "success": False
        }

        try:
            crawler = HashnodeCrawler()
            articles = await crawler.crawl()
            saved = await self._process_and_save_articles(articles)

            stats["crawled"] = len(articles)
            stats["saved"] = saved
            stats["success"] = True

        except Exception as e:
            logger.error(f"Error crawling Hashnode: {e}", exc_info=True)
            stats["error"] = str(e)

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Hashnode crawler completed. Saved: {stats['saved']}")

        return stats

    # async def run_medium_crawler(self) -> Dict[str, Any]:
    #     """
    #     Run Medium crawler
    #     Disabled: ~70% of articles are RSS excerpts ("Continue reading on..."), not full content.
    #     """
    #     logger.info("Starting Medium crawler...")
    #
    #     stats = {
    #         "started_at": datetime.utcnow().isoformat(),
    #         "source": "medium",
    #         "crawled": 0,
    #         "saved": 0,
    #         "success": False
    #     }
    #
    #     try:
    #         crawler = MediumCrawler()
    #         articles = await crawler.crawl()
    #         saved = await self._process_and_save_articles(articles)
    #
    #         stats["crawled"] = len(articles)
    #         stats["saved"] = saved
    #         stats["success"] = True
    #
    #     except Exception as e:
    #         logger.error(f"Error crawling Medium: {e}", exc_info=True)
    #         stats["error"] = str(e)
    #
    #     stats["completed_at"] = datetime.utcnow().isoformat()
    #     logger.info(f"Medium crawler completed. Saved: {stats['saved']}")
    #
    #     return stats

    async def run_reddit_crawler(self) -> Dict[str, Any]:
        """
        Run Reddit crawler

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting Reddit crawler...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "source": "reddit",
            "crawled": 0,
            "saved": 0,
            "success": False
        }

        try:
            crawler = RedditCrawler(known_url_filter=self._known_article_urls)
            articles = await crawler.crawl()
            saved = await self._process_and_save_articles(articles)

            stats["crawled"] = len(articles)
            stats["saved"] = saved
            stats["success"] = True

        except Exception as e:
            logger.error(f"Error crawling Reddit: {e}", exc_info=True)
            stats["error"] = str(e)

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Reddit crawler completed. Saved: {stats['saved']}")

        return stats

    async def run_hackernews_crawler(self) -> Dict[str, Any]:
        """
        Run Hacker News crawler

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting Hacker News crawler...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "source": "hackernews",
            "crawled": 0,
            "saved": 0,
            "success": False
        }

        try:
            crawler = HackerNewsCrawler(known_url_filter=self._known_article_urls)
            articles = await crawler.crawl()
            saved = await self._process_and_save_articles(articles)

            stats["crawled"] = len(articles)
            stats["saved"] = saved
            stats["success"] = True

        except Exception as e:
            logger.error(f"Error crawling Hacker News: {e}", exc_info=True)
            stats["error"] = str(e)

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Hacker News crawler completed. Saved: {stats['saved']}")

        return stats

    async def _process_and_save_articles(self, articles: List[RawArticle]) -> int:
        """
        Process raw articles through the pipeline and save to database

        Pipeline:
        1. Deduplicate (URL)
        2. Content gate — drop empty / blocked / too-short bodies (no LLM call)
        3. Per article: LLM triage (is it a real article, developer-relevant,
           category, tags, Korean title) → Korean translation → validation
        4. Score and save each article as soon as its summary is ready, so a
           Lambda timeout loses only in-flight articles

        Args:
            articles: List of RawArticles to process

        Returns:
            Number of articles saved
        """
        if not articles:
            logger.info("No articles to process")
            return 0

        logger.info(f"Processing {len(articles)} articles...")

        db = SessionLocal()
        try:
            # Step 1: Deduplicate
            deduplicator = DeduplicatorService(db)
            unique_articles = deduplicator.filter_duplicates(articles, check_db=True)
            logger.info(f"After deduplication: {len(unique_articles)} unique articles")

            if not unique_articles:
                return 0

            # Step 2: Content gate. Translating an empty or junk body is how the
            # LLM ended up inventing articles, so these never reach it.
            articles_to_summarize = []
            gate_rejects: Counter = Counter()
            for article in unique_articles:
                check = assess_content(article.content or "", settings.MIN_ARTICLE_CONTENT_CHARS)
                if check.ok:
                    articles_to_summarize.append(article)
                else:
                    gate_rejects[check.reason.split(":")[0]] += 1
                    logger.info(f"Content gate rejected ({check.reason}): {article.url}")

            logger.info(
                f"Content gate: {len(articles_to_summarize)}/{len(unique_articles)} passed, "
                f"rejected={dict(gate_rejects)}"
            )
            if not articles_to_summarize:
                return 0

            # Steps 3–4: summarize concurrently, save each result immediately
            saved_count = 0
            outcomes: Counter = Counter()
            async for article, result in self.summarizer.iter_summaries(articles_to_summarize):
                outcomes[result.status] += 1
                if result.status != "ok":
                    logger.info(f"Not saved ({result.status}: {result.reason}): {article.url}")
                    continue
                if self._save_article(db, article, result):
                    saved_count += 1

            logger.info(f"Summaries: {dict(outcomes)}. Saved {saved_count} articles to database")
            return saved_count

        except Exception as e:
            logger.error(f"Error processing articles: {e}", exc_info=True)
            db.rollback()
            return 0

        finally:
            db.close()

    def _save_article(self, db: Session, article: RawArticle, summary: SummaryResult) -> bool:
        """Insert one article and its tags in its own transaction."""
        if article.source == "github":
            item_type = ItemType.REPO
        elif article.source == "reddit" or article.raw_data.get("hn_id"):
            item_type = ItemType.DISCUSSION
        else:
            item_type = ItemType.BLOG

        # Prefer tags from LLM; fallback to article tags
        tags = summary.tags or article.tags
        now = datetime.utcnow()

        try:
            db_article = Article(
                # Always a fresh UUID (matches Java backend's UUID.randomUUID())
                external_id=str(uuid.uuid4()),
                item_type=item_type,
                source=article.source,
                category=self._normalize_category(summary.category),
                summary_ko_title=summary.title_ko or article.title_en[:100],
                summary_ko_body=summary.summary_ko,
                title_en=article.title_en,
                url=article.url,
                score=self.scorer.calculate_score(article),
                stars=article.stars,
                comments=article.comments,
                upvotes=article.upvotes,
                read_time=article.read_time,
                language=article.language,
                created_at_source=article.published_at,
                created_at=now,
                updated_at=now,
            )
            db.add(db_article)
            db.flush()  # Flush to get the article ID
            for tag in tags or []:
                db.merge(ArticleTag(article_id=db_article.id, tag=tag))
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save article {article.url}: {e}")
            return False

    @staticmethod
    def _known_article_urls(urls: List[str]) -> set:
        """Subset of ``urls`` already stored in the articles table."""
        if not urls:
            return set()
        db = SessionLocal()
        try:
            rows = db.query(Article.url).filter(Article.url.in_(set(urls))).all()
            return {row[0] for row in rows}
        finally:
            db.close()

    async def _send_short_content_webhook(self, articles: list) -> None:
        """Send short/empty-content articles to Discord for manual admin review."""
        source_groups: dict[str, list[dict]] = {}
        for article in articles:
            raw = article.raw_data
            if raw.get("hn_id"):
                source_name = "HackerNews"
                discussion_url = raw.get("hn_discussion_url")
            elif raw.get("permalink"):
                source_name = "Reddit"
                discussion_url = f"https://www.reddit.com{raw['permalink']}"
            else:
                source_name = article.source.capitalize()
                discussion_url = None

            entry = {
                "title": article.title_en,
                "url": article.url,
                "discussion_url": discussion_url,
                "upvotes": article.upvotes,
                "comments": article.comments,
            }
            source_groups.setdefault(source_name, []).append(entry)

        for source_name, entries in source_groups.items():
            await BaseCrawler.send_discord_webhook(source_name, entries)

    async def _process_and_save_repositories(self, repos: List[RawArticle]) -> int:
        """
        Mirror github.com/trending into the git_repos table.

        Snapshot semantics: after this call, git_repos equals the set of
        repos returned by the scrape. New URLs are summarized+inserted,
        existing URLs have metrics refreshed, and rows no longer on
        trending are removed.

        Returns the number of rows inserted + updated.
        """
        # Empty-crawl guard: do NOT wipe the table if the scraper returned
        # nothing (HTML change, network failure, GitHub block).
        if not repos:
            logger.info("No repositories crawled — skipping DB update to preserve existing snapshot")
            return 0

        # Intra-batch dedup by URL, keep first occurrence
        crawled_by_url: Dict[str, RawArticle] = {}
        for r in repos:
            crawled_by_url.setdefault(r.url, r)
        crawled_urls = set(crawled_by_url.keys())

        logger.info(f"Processing {len(crawled_by_url)} trending repositories...")

        db = SessionLocal()
        try:
            existing_rows = {
                row.url: row
                for row in db.query(GitRepo).filter(GitRepo.url.in_(crawled_urls)).all()
            }

            new_repos = [r for url, r in crawled_by_url.items() if url not in existing_rows]
            existing_pairs = [(existing_rows[url], crawled_by_url[url]) for url in existing_rows]

            logger.info(
                f"Snapshot diff: new={len(new_repos)}, existing={len(existing_pairs)}"
            )

            # Summarize only new URLs (costly LLM call), grounded in each README
            results: Dict[str, SummaryResult] = {}
            if new_repos:
                await GitHubCrawler().attach_readmes(new_repos)
                async for repo, result in self.summarizer.iter_summaries(new_repos, kind="repo"):
                    results[repo.url] = result

            inserted = 0
            failed_summary = 0
            non_technical = 0
            for repo in new_repos:
                summary = results.get(repo.url)
                if summary is None or summary.status != "ok":
                    if summary is not None and summary.status == "non_technical":
                        non_technical += 1
                    else:
                        failed_summary += 1
                    continue

                category = self._normalize_category(summary.category)
                score = self.scorer.calculate_score(repo)

                db.add(GitRepo(
                    full_name=repo.title_en,
                    url=repo.url,
                    description=repo.content,
                    language=repo.language,
                    stars=repo.stars or 0,
                    forks=repo.raw_data.get("forks", 0),
                    stars_this_week=repo.raw_data.get("stars_this_week", 0),
                    summary_ko_title=summary.title_ko or repo.title_en[:100],
                    summary_ko_body=summary.summary_ko,
                    category=category,
                    score=score,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                ))
                inserted += 1

            # Refresh metrics on existing rows; preserve Korean summary + category
            updated = 0
            now = datetime.utcnow()
            for row, fresh in existing_pairs:
                row.stars = fresh.stars or 0
                row.forks = fresh.raw_data.get("forks", 0)
                row.stars_this_week = fresh.raw_data.get("stars_this_week", 0)
                if fresh.content:
                    row.description = fresh.content
                if fresh.language:
                    row.language = fresh.language
                if fresh.title_en:
                    row.full_name = fresh.title_en
                row.score = self.scorer.calculate_score(fresh)
                row.updated_at = now
                updated += 1

            # Remove rows that are no longer on trending
            deleted = db.query(GitRepo).filter(
                ~GitRepo.url.in_(crawled_urls)
            ).delete(synchronize_session=False)

            db.commit()
            logger.info(
                f"git_repos snapshot updated: inserted={inserted}, updated={updated}, "
                f"deleted={deleted}, skipped_failed_summary={failed_summary}, "
                f"skipped_non_technical={non_technical}"
            )
            return inserted + updated

        except Exception as e:
            logger.error(f"Error processing repositories: {e}", exc_info=True)
            db.rollback()
            return 0

        finally:
            db.close()

    async def run_deduplication(self) -> Dict[str, Any]:
        """
        Run deduplication on existing database articles

        Returns:
            Dictionary with deduplication statistics
        """
        logger.info("Starting deduplication...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "removed": 0,
            "success": False
        }

        db = SessionLocal()
        try:
            deduplicator = DeduplicatorService(db)
            removed_count = deduplicator.mark_existing_duplicates()

            stats["removed"] = removed_count
            stats["success"] = True

        except Exception as e:
            logger.error(f"Error during deduplication: {e}", exc_info=True)
            stats["error"] = str(e)

        finally:
            db.close()

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Deduplication completed. Removed: {stats['removed']}")

        return stats

    async def refresh_scores(self, days: int = None) -> Dict[str, Any]:
        """
        Recalculate scores for articles within the scoring window

        This updates scores to reflect time decay as articles age.
        Only processes articles within the max age threshold since older
        articles will always have score=0 anyway.

        Args:
            days: Number of days back to refresh scores (default: use SCORE_MAX_AGE_DAYS from settings)

        Returns:
            Dictionary with refresh statistics
        """
        from app.config.settings import settings
        from datetime import timedelta

        # Use SCORE_MAX_AGE_DAYS if days not specified (more efficient)
        if days is None:
            days = getattr(settings, 'SCORE_MAX_AGE_DAYS', 14)

        logger.info(f"Starting score refresh for articles from last {days} days...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "updated": 0,
            "zeroed": 0,
            "success": False
        }

        db = SessionLocal()
        try:
            # Get articles from last N days (only articles within scoring window)
            cutoff = datetime.utcnow() - timedelta(days=days)
            articles = db.query(Article).filter(
                Article.created_at_source >= cutoff
            ).all()

            logger.info(f"Found {len(articles)} articles to refresh (within {days}-day window)")

            # Recalculate each score
            for article in articles:
                try:
                    # Convert DB Article to RawArticle for scoring
                    raw_article = RawArticle(
                        title_en=article.title_en,
                        url=article.url,
                        source=article.source,
                        published_at=article.created_at_source,
                        tags=[],  # Not needed for scoring
                        content="",  # Not needed for scoring
                        stars=article.stars,
                        upvotes=article.upvotes,
                        comments=article.comments,
                        read_time=article.read_time,
                        language=article.language
                    )

                    # Recalculate score with current time decay
                    old_score = article.score
                    new_score = self.scorer.calculate_score(raw_article)
                    article.score = new_score
                    stats["updated"] += 1

                    # Track articles that got zeroed out
                    if new_score == 0 and old_score > 0:
                        stats["zeroed"] += 1

                except Exception as e:
                    logger.warning(f"Failed to refresh score for article {article.id}: {e}")
                    continue

            # Commit all updates
            db.commit()
            stats["success"] = True

        except Exception as e:
            logger.error(f"Error during score refresh: {e}", exc_info=True)
            stats["error"] = str(e)
            db.rollback()

        finally:
            db.close()

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"Score refresh completed. Updated: {stats['updated']} articles, "
            f"Zeroed out: {stats['zeroed']} articles (14+ days old)"
        )

        return stats

    @staticmethod
    def _normalize_category(category: str) -> str:
        """
        Ensure category matches allowed enum values, otherwise fallback to OTHER.
        """
        allowed = {c.value for c in Category}
        return category if category in allowed else Category.OTHER.value
