"""Deduplication service to prevent duplicate articles"""

from typing import List, Set
from difflib import SequenceMatcher
from sqlalchemy.orm import Session
from app.models.article import Article
from app.crawlers.base import RawArticle
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class DeduplicatorService:
    """Service to detect and prevent duplicate articles"""

    def __init__(self, db: Session):
        self.db = db
        self.similarity_threshold = settings.TITLE_SIMILARITY_THRESHOLD

    def is_duplicate(self, article: RawArticle) -> bool:
        """
        Check if an article is a duplicate

        First checks URL, then title similarity

        Args:
            article: RawArticle to check

        Returns:
            True if duplicate, False otherwise
        """
        # Check 1: Exact URL match
        if self._url_exists(article.url):
            logger.debug(f"Duplicate URL found: {article.url}")
            return True

        # Check 2: Title similarity
        if self._similar_title_exists(article.title_en):
            logger.debug(f"Similar title found: {article.title_en[:50]}")
            return True

        return False

    def _url_exists(self, url: str) -> bool:
        """Check if URL already exists in database"""
        existing = self.db.query(Article).filter(Article.url == url).first()
        return existing is not None

    def _similar_title_exists(self, title: str) -> bool:
        """
        Check if a similar title exists in database

        Uses fuzzy string matching to find similar titles

        Args:
            title: Title to check

        Returns:
            True if similar title exists
        """
        # Get recent articles (last 30 days) to check against
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)

        recent_articles = self.db.query(Article).filter(
            Article.created_at >= thirty_days_ago
        ).all()

        for existing in recent_articles:
            similarity = self._calculate_similarity(title, existing.title_en)
            if similarity >= self.similarity_threshold:
                logger.debug(
                    f"Similar title match ({similarity:.2f}): "
                    f"'{title[:50]}' ~ '{existing.title_en[:50]}'"
                )
                return True

        return False

    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        # Normalize strings
        s1 = str1.lower().strip()
        s2 = str2.lower().strip()

        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, s1, s2).ratio()

    def filter_duplicates(
        self,
        articles: List[RawArticle],
        check_db: bool = True
    ) -> List[RawArticle]:
        """
        Filter out duplicate articles from a list

        Args:
            articles: List of RawArticles to filter
            check_db: Whether to check database for duplicates

        Returns:
            List of non-duplicate articles
        """
        unique_articles = []
        seen_urls = set()
        seen_titles = set()

        for article in articles:
            # Skip if URL already seen in this batch
            if article.url in seen_urls:
                logger.debug(f"Duplicate in batch (URL): {article.url}")
                continue

            # Skip if similar title already seen in this batch
            is_similar_in_batch = any(
                self._calculate_similarity(article.title_en, title) >= self.similarity_threshold
                for title in seen_titles
            )
            if is_similar_in_batch:
                logger.debug(f"Duplicate in batch (title): {article.title_en[:50]}")
                continue

            # Check database if requested
            if check_db and self.is_duplicate(article):
                logger.debug(f"Duplicate in database: {article.url}")
                continue

            # Article is unique
            unique_articles.append(article)
            seen_urls.add(article.url)
            seen_titles.add(article.title_en)

        logger.info(
            f"Filtered {len(articles)} articles -> {len(unique_articles)} unique"
        )

        return unique_articles

    def mark_existing_duplicates(self) -> int:
        """
        Find and mark existing duplicate articles in database

        This is a maintenance operation to clean up the database

        Returns:
            Number of duplicates found and removed
        """
        all_articles = self.db.query(Article).order_by(Article.created_at.desc()).all()

        seen_urls = set()
        seen_titles = set()
        duplicates_to_remove = []

        for article in all_articles:
            # Check URL
            if article.url in seen_urls:
                duplicates_to_remove.append(article.id)
                continue

            # Check title similarity
            is_similar = any(
                self._calculate_similarity(article.title_en, title) >= self.similarity_threshold
                for title in seen_titles
            )
            if is_similar:
                duplicates_to_remove.append(article.id)
                continue

            # Mark as seen
            seen_urls.add(article.url)
            seen_titles.add(article.title_en)

        # Remove duplicates
        if duplicates_to_remove:
            self.db.query(Article).filter(
                Article.id.in_(duplicates_to_remove)
            ).delete(synchronize_session=False)
            self.db.commit()

        logger.info(f"Removed {len(duplicates_to_remove)} duplicate articles")
        return len(duplicates_to_remove)
