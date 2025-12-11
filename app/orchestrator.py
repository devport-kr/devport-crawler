"""Orchestrator to coordinate crawler execution and data processing"""

from typing import List, Dict, Any
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from app.crawlers.devto import DevToCrawler
from app.crawlers.hashnode import HashnodeCrawler
from app.crawlers.medium import MediumCrawler
from app.crawlers.reddit import RedditCrawler
from app.crawlers.github import GitHubCrawler
from app.crawlers.llm_rankings import LLMRankingsCrawler
from app.crawlers.base import RawArticle
from app.services.summarizer import SummarizerService
from app.services.scorer import ScorerService
from app.services.deduplicator import DeduplicatorService
from app.models.article import Article, ItemType, Category
from app.models.article_tag import ArticleTag
from app.models.git_repo import GitRepo
from app.config.database import SessionLocal

logger = logging.getLogger(__name__)


class CrawlerOrchestrator:
    """Orchestrates the crawling, processing, and storage of articles"""

    def __init__(self):
        self.summarizer = SummarizerService()
        self.scorer = ScorerService()

    async def run_all_crawlers(self) -> Dict[str, Any]:
        """
        Run all blog crawlers sequentially

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting all blog crawlers...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "sources": {},
            "total_crawled": 0,
            "total_saved": 0,
            "errors": []
        }

        # Run each crawler
        sources = [
            ("devto", DevToCrawler()),
            ("hashnode", HashnodeCrawler()),
            ("medium", MediumCrawler()),
            ("reddit", RedditCrawler())
        ]

        for source_name, crawler in sources:
            try:
                logger.info(f"Running {source_name} crawler...")
                articles = await crawler.crawl()
                saved = await self._process_and_save_articles(articles)

                stats["sources"][source_name] = {
                    "crawled": len(articles),
                    "saved": saved,
                    "success": True
                }
                stats["total_crawled"] += len(articles)
                stats["total_saved"] += saved

            except Exception as e:
                logger.error(f"Error crawling {source_name}: {e}", exc_info=True)
                stats["sources"][source_name] = {
                    "crawled": 0,
                    "saved": 0,
                    "success": False,
                    "error": str(e)
                }
                stats["errors"].append(f"{source_name}: {str(e)}")

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"All crawlers completed. Total saved: {stats['total_saved']}")

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

    async def run_medium_crawler(self) -> Dict[str, Any]:
        """
        Run Medium crawler

        Returns:
            Dictionary with crawling statistics
        """
        logger.info("Starting Medium crawler...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "source": "medium",
            "crawled": 0,
            "saved": 0,
            "success": False
        }

        try:
            crawler = MediumCrawler()
            articles = await crawler.crawl()
            saved = await self._process_and_save_articles(articles)

            stats["crawled"] = len(articles)
            stats["saved"] = saved
            stats["success"] = True

        except Exception as e:
            logger.error(f"Error crawling Medium: {e}", exc_info=True)
            stats["error"] = str(e)

        stats["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Medium crawler completed. Saved: {stats['saved']}")

        return stats

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
            crawler = RedditCrawler()
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

    async def _process_and_save_articles(self, articles: List[RawArticle]) -> int:
        """
        Process raw articles through the pipeline and save to database

        Pipeline:
        1. Deduplicate
        2. Summarize (Korean) and Categorize (LLM does both)
        3. Calculate score
        4. Save to database

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

            # Step 2: Summarize and Categorize (Korean) - LLM does both now
            # Efficient batching: 25 articles per LLM request, 10 seconds between requests
            summaries = await self.summarizer.summarize_batch(unique_articles)
            logger.info("Summarization and categorization completed")

            # Step 3: Filter out failed summarizations and calculate scores
            scored_articles = []
            failed_count = 0
            for article, summary in zip(unique_articles, summaries):
                if summary is None:
                    failed_count += 1
                    logger.warning(f"Skipping article due to failed summarization: {article.title_en}")
                    continue

                # Get category from LLM response
                category = self._normalize_category(summary.get("category", "OTHER"))
                score = self.scorer.calculate_score(article)
                scored_articles.append((article, category, summary, score))

            logger.info(f"Scoring completed. {failed_count} articles skipped due to failed summarization")

            # Step 5: Save to database
            saved_count = 0
            for article, category, summary, score in scored_articles:
                try:
                    # Determine item type
                    item_type = ItemType.REPO if article.source == "github" else ItemType.BLOG

                    # Prefer tags from LLM; fallback to article tags
                    tags = summary.get("tags") if summary else None
                    tags = tags if tags else article.tags

                    # Create Article model
                    db_article = Article(
                        item_type=item_type,
                        source=article.source,
                        category=category,
                        summary_ko_title=summary.get("title_ko", article.title_en[:100]),
                        summary_ko_body=summary.get("summary_ko"),
                        title_en=article.title_en,
                        url=article.url,
                        score=score,
                        # tags=article.tags,  # NOTE: Tags stored in separate table
                        stars=article.stars,
                        comments=article.comments,
                        upvotes=article.upvotes,
                        read_time=article.read_time,
                        language=article.language,
                        created_at_source=article.published_at,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )

                    db.add(db_article)
                    db.flush()  # Flush to get the article ID

                    # Save tags to article_tags table if article has tags
                    if tags:
                        for tag in tags:
                            tag_entry = ArticleTag(
                                article_id=db_article.id,
                                tag=tag
                            )
                            db.merge(tag_entry)  # Use merge to avoid identity map conflicts

                    saved_count += 1

                except Exception as e:
                    logger.error(f"Failed to save article: {e}")
                    db.rollback()
                    continue

            # Commit all articles and tags
            db.commit()
            logger.info(f"Saved {saved_count} articles to database")

            return saved_count

        except Exception as e:
            logger.error(f"Error processing articles: {e}", exc_info=True)
            db.rollback()
            return 0

        finally:
            db.close()

    async def _process_and_save_repositories(self, repos: List[RawArticle]) -> int:
        """
        Process GitHub repos and save to git_repos table

        Pipeline:
        1. Deduplicate by URL
        2. Summarize (Korean title + description) and Categorize (LLM does both)
        3. Calculate score
        4. Save to git_repos table

        Args:
            repos: List of RawArticles representing GitHub repos

        Returns:
            Number of repositories saved
        """
        if not repos:
            logger.info("No repositories to process")
            return 0

        logger.info(f"Processing {len(repos)} repositories...")

        db = SessionLocal()
        try:
            # Step 1: Deduplicate by URL
            existing_urls = {r.url for r in db.query(GitRepo.url).all()}
            unique_repos = [r for r in repos if r.url not in existing_urls]
            logger.info(f"After deduplication: {len(unique_repos)} unique repositories")

            if not unique_repos:
                return 0

            # Step 2: Summarize and Categorize (Korean) - LLM does both
            summaries = await self.summarizer.summarize_batch(unique_repos)
            logger.info("Summarization and categorization completed")

            # Step 3: Filter out failed summarizations and calculate scores
            scored_repos = []
            failed_count = 0
            for repo, summary in zip(unique_repos, summaries):
                if summary is None:
                    failed_count += 1
                    logger.warning(f"Skipping repo due to failed summarization: {repo.title_en}")
                    continue

                # Get category from LLM response
                category = self._normalize_category(summary.get("category", "OTHER"))
                score = self.scorer.calculate_score(repo)
                scored_repos.append((repo, category, summary, score))

            logger.info(f"Scoring completed. {failed_count} repos skipped due to failed summarization")

            # Step 4: Save to git_repos table
            saved_count = 0
            for repo, category, summary, score in scored_repos:
                try:
                    # Create GitRepo model
                    db_repo = GitRepo(
                        full_name=repo.title_en,
                        url=repo.url,
                        description=repo.content,
                        language=repo.language,
                        stars=repo.stars or 0,
                        forks=repo.raw_data.get("forks", 0),
                        stars_this_week=repo.raw_data.get("stars_this_week", 0),
                        summary_ko_title=summary.get("title_ko", repo.title_en[:100]),
                        summary_ko_body=summary.get("summary_ko"),
                        category=category,
                        score=score,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )

                    db.add(db_repo)
                    saved_count += 1

                except Exception as e:
                    logger.error(f"Failed to save repository: {e}")
                    db.rollback()
                    continue

            # Commit all repositories
            db.commit()
            logger.info(f"Saved {saved_count} repositories to database")

            return saved_count

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

    async def refresh_scores(self, days: int = 30) -> Dict[str, Any]:
        """
        Recalculate scores for articles from last N days

        This updates scores to reflect time decay as articles age.

        Args:
            days: Number of days back to refresh scores (default: 30)

        Returns:
            Dictionary with refresh statistics
        """
        logger.info(f"Starting score refresh for articles from last {days} days...")

        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "updated": 0,
            "success": False
        }

        db = SessionLocal()
        try:
            from datetime import timedelta

            # Get articles from last N days
            cutoff = datetime.utcnow() - timedelta(days=days)
            articles = db.query(Article).filter(
                Article.created_at_source >= cutoff
            ).all()

            logger.info(f"Found {len(articles)} articles to refresh")

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
                    new_score = self.scorer.calculate_score(raw_article)
                    article.score = new_score
                    stats["updated"] += 1

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
        logger.info(f"Score refresh completed. Updated: {stats['updated']} articles")

        return stats

    @staticmethod
    def _normalize_category(category: str) -> str:
        """
        Ensure category matches allowed enum values, otherwise fallback to OTHER.
        """
        allowed = {c.value for c in Category}
        return category if category in allowed else Category.OTHER.value
