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

        elif article.source in ["devto", "medium"]:
            # For blogs, use upvotes/reactions
            return article.upvotes or 0

        else:
            # Default fallback
            return max(article.stars or 0, article.upvotes or 0)

    @staticmethod
    def _calculate_time_decay(published_at: datetime) -> float:
        """
        Calculate time decay multiplier using exponential decay

        Three-phase decay strategy:
        - Phase 1 (0-2 days): 2.0x plateau - Fresh content stays on top
        - Phase 2 (2-14 days): Exponential decay - Gradual decline with 4-day half-life
        - Phase 3 (14+ days): 0.0x zero out - Old content completely buried

        The exponential decay formula: 2.0 * e^(-(t-2)/4)
        - Day 0-2:  2.00x (plateau)
        - Day 4:    1.21x (starting to age)
        - Day 7:    0.70x (declining)
        - Day 10:   0.41x (fading)
        - Day 14:   0.20x (almost gone)
        - Day 14+:  0.00x (buried)

        This ensures:
        - Recent articles dominate the dashboard for 2 days
        - Scores gradually decrease over the next 12 days
        - Articles older than 2 weeks are effectively hidden

        Args:
            published_at: When the article was published

        Returns:
            Time decay multiplier (0.0 to 2.0)
        """
        now = datetime.utcnow()
        if published_at.tzinfo:
            # Make now timezone-aware if published_at is
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)

        age = now - published_at
        age_days = age.total_seconds() / 86400  # Use fractional days for smoother decay

        # Get configurable parameters (with defaults)
        plateau_days = getattr(settings, 'SCORE_PLATEAU_DAYS', 2)
        half_life_days = getattr(settings, 'SCORE_HALF_LIFE_DAYS', 4.0)
        max_age_days = getattr(settings, 'SCORE_MAX_AGE_DAYS', 14)

        # Phase 1: Plateau - Fresh content stays at maximum boost
        if age_days <= plateau_days:
            return 2.0

        # Phase 2: Exponential decay - Smooth gradual decline
        if age_days <= max_age_days:
            decay_time = age_days - plateau_days
            multiplier = 2.0 * math.exp(-decay_time / half_life_days)
            return multiplier

        # Phase 3: Zero out - Old content gets no visibility
        return 0.0

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
