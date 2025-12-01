"""Score calculation service for articles"""

from datetime import datetime, timedelta
import math
from app.crawlers.base import RawArticle
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class ScorerService:
    """Service to calculate unified scores for articles across different sources"""

    @staticmethod
    def calculate_score(article: RawArticle) -> int:
        """
        Calculate a unified score for an article

        The score combines:
        - Base engagement (stars/upvotes/reactions)
        - Comment count
        - Time decay (newer = higher)
        - Source weight

        Args:
            article: RawArticle to score

        Returns:
            Integer score (higher is better)
        """
        # Get base engagement score
        base_score = ScorerService._get_base_engagement(article)

        # Apply time decay
        time_multiplier = ScorerService._calculate_time_decay(article.published_at)

        # Apply source weight
        source_weight = ScorerService._get_source_weight(article.source)

        # Apply comment multiplier (discussions are valuable)
        comment_multiplier = ScorerService._calculate_comment_multiplier(
            article.comments or 0
        )

        # Calculate final score
        final_score = int(
            base_score * time_multiplier * source_weight * comment_multiplier
        )

        logger.debug(
            f"Scored '{article.title_en[:50]}': "
            f"base={base_score}, time={time_multiplier:.2f}, "
            f"source={source_weight}, comments={comment_multiplier:.2f}, "
            f"final={final_score}"
        )

        return max(1, final_score)  # Minimum score of 1

    @staticmethod
    def _get_base_engagement(article: RawArticle) -> int:
        """Get base engagement score from article metrics"""
        if article.source == "github":
            # For GitHub, stars are the primary metric
            return article.stars or 0

        elif article.source in ["devto", "hashnode", "medium"]:
            # For blogs, use upvotes/reactions
            return article.upvotes or 0

        else:
            # Default fallback
            return max(article.stars or 0, article.upvotes or 0)

    @staticmethod
    def _calculate_time_decay(published_at: datetime) -> float:
        """
        Calculate time decay multiplier

        Newer content gets higher scores:
        - 0-1 days: 2.0x
        - 1-3 days: 1.5x
        - 3-7 days: 1.0x
        - 7-14 days: 0.7x
        - 14-30 days: 0.5x
        - 30+ days: 0.3x

        Args:
            published_at: When the article was published

        Returns:
            Time decay multiplier
        """
        now = datetime.utcnow()
        if published_at.tzinfo:
            # Make now timezone-aware if published_at is
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)

        age = now - published_at
        days = age.days

        if days < 1:
            return 2.0
        elif days < 3:
            return 1.5
        elif days < 7:
            return 1.0
        elif days < 14:
            return 0.7
        elif days < 30:
            return 0.5
        else:
            return 0.3

    @staticmethod
    def _get_source_weight(source: str) -> float:
        """
        Get weight multiplier for different sources

        GitHub repos are weighted higher as they represent
        more substantial content.

        Args:
            source: Source name

        Returns:
            Source weight multiplier
        """
        if source == "github":
            return settings.GITHUB_SOURCE_WEIGHT
        else:
            return settings.BLOG_SOURCE_WEIGHT

    @staticmethod
    def _calculate_comment_multiplier(comments: int) -> float:
        """
        Calculate multiplier based on comment count

        More comments = more engagement = higher multiplier

        Args:
            comments: Number of comments

        Returns:
            Comment multiplier (1.0 to 1.5)
        """
        if comments == 0:
            return 1.0
        elif comments < 5:
            return 1.1
        elif comments < 10:
            return 1.2
        elif comments < 20:
            return 1.3
        elif comments < 50:
            return 1.4
        else:
            return 1.5

    @staticmethod
    def normalize_scores(articles: list[tuple[RawArticle, int]]) -> list[tuple[RawArticle, int]]:
        """
        Normalize scores to a 0-1000 range (optional post-processing)

        Args:
            articles: List of (article, score) tuples

        Returns:
            List of (article, normalized_score) tuples
        """
        if not articles:
            return articles

        scores = [score for _, score in articles]
        max_score = max(scores)
        min_score = min(scores)

        if max_score == min_score:
            return articles

        normalized = []
        for article, score in articles:
            # Normalize to 0-1000 range
            normalized_score = int(
                ((score - min_score) / (max_score - min_score)) * 1000
            )
            normalized.append((article, normalized_score))

        return normalized
