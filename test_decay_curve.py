#!/usr/bin/env python3
"""
Test script to validate the exponential time decay curve

Generates a table showing the decay multiplier for articles at different ages.
This helps verify that the scoring behavior matches the requirements:
- Recent articles (0-2 days): Full boost (2.0x)
- Aging articles: Gradual decay every 2 days
- Old articles (14+ days): Near-zero or zero score
"""

from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.scorer import ScorerService
from app.config.settings import settings


def test_decay_curve():
    """Test the time decay curve with various article ages"""

    print("=" * 70)
    print("TIME DECAY CURVE VALIDATION")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  - Plateau Days:   {getattr(settings, 'SCORE_PLATEAU_DAYS', 2)}")
    print(f"  - Half-Life Days: {getattr(settings, 'SCORE_HALF_LIFE_DAYS', 4.0)}")
    print(f"  - Max Age Days:   {getattr(settings, 'SCORE_MAX_AGE_DAYS', 14)}")
    print()
    print("=" * 70)
    print(f"{'Age (days)':<12} | {'Multiplier':<12} | {'Status':<30}")
    print("=" * 70)

    base_time = datetime.utcnow()

    # Test key milestones
    test_ages = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30]

    for days in test_ages:
        published = base_time - timedelta(days=days)
        multiplier = ScorerService._calculate_time_decay(published)

        # Determine status
        if days <= 2:
            status = "Top of dashboard (fresh)"
        elif days <= 7:
            status = "Starting to age"
        elif days < 14:
            status = "Declining"
        elif days == 14:
            status = "Almost gone"
        else:
            status = "Buried (zero score)"

        print(f"{days:<12} | {multiplier:<12.2f} | {status:<30}")

    print("=" * 70)
    print()

    # Verify requirements
    print("REQUIREMENT VERIFICATION:")
    print()

    # Requirement 1: Recent articles (0-2 days) have full/high score
    day_0 = ScorerService._calculate_time_decay(base_time)
    day_2 = ScorerService._calculate_time_decay(base_time - timedelta(days=2))
    req1_pass = day_0 == 2.0 and day_2 == 2.0
    print(f"✓ Req 1: Recent articles (0-2 days) stay on top" if req1_pass else "✗ Req 1: FAILED")
    print(f"  Day 0: {day_0:.2f}x, Day 2: {day_2:.2f}x (expected: 2.00x for both)")
    print()

    # Requirement 2: Score decreases every 2 days
    day_4 = ScorerService._calculate_time_decay(base_time - timedelta(days=4))
    day_7 = ScorerService._calculate_time_decay(base_time - timedelta(days=7))
    day_10 = ScorerService._calculate_time_decay(base_time - timedelta(days=10))
    req2_pass = day_2 > day_4 > day_7 > day_10
    print(f"✓ Req 2: Scores decrease gradually" if req2_pass else "✗ Req 2: FAILED")
    print(f"  Day 2: {day_2:.2f}x → Day 4: {day_4:.2f}x → Day 7: {day_7:.2f}x → Day 10: {day_10:.2f}x")
    print()

    # Requirement 3: Old articles (14+ days) near-zero or zero
    day_14 = ScorerService._calculate_time_decay(base_time - timedelta(days=14))
    day_15 = ScorerService._calculate_time_decay(base_time - timedelta(days=15))
    day_30 = ScorerService._calculate_time_decay(base_time - timedelta(days=30))
    req3_pass = day_14 <= 0.25 and day_15 == 0.0 and day_30 == 0.0
    print(f"✓ Req 3: Old articles (14+ days) buried" if req3_pass else "✗ Req 3: FAILED")
    print(f"  Day 14: {day_14:.2f}x, Day 15: {day_15:.2f}x, Day 30: {day_30:.2f}x (expected: ≤0.25, 0.00, 0.00)")
    print()

    # Overall
    all_pass = req1_pass and req2_pass and req3_pass
    print("=" * 70)
    if all_pass:
        print("✓ ALL REQUIREMENTS PASSED")
    else:
        print("✗ SOME REQUIREMENTS FAILED - Review configuration")
    print("=" * 70)
    print()

    return all_pass


def test_with_base_score():
    """Test actual score calculations with a sample article"""

    print()
    print("=" * 70)
    print("SCORE CALCULATION EXAMPLES")
    print("=" * 70)
    print()
    print("Simulating a GitHub repo with 100 stars, 5 comments:")
    print()
    print(f"{'Age':<12} | {'Time Decay':<12} | {'Final Score':<12} | {'Rank Impact'}")
    print("=" * 70)

    base_engagement = 100
    source_weight = 2.0  # GitHub
    comment_multiplier = 1.2  # 5 comments

    base_time = datetime.utcnow()

    for days in [0, 2, 4, 7, 10, 14, 15]:
        published = base_time - timedelta(days=days)
        time_decay = ScorerService._calculate_time_decay(published)
        final_score = int(base_engagement * time_decay * source_weight * comment_multiplier)

        if days <= 2:
            impact = "Top of dashboard"
        elif days <= 7:
            impact = "Visible on first page"
        elif days < 14:
            impact = "Pushed down"
        else:
            impact = "Invisible (buried)"

        print(f"{days} days<4} | {time_decay:<12.2f} | {final_score:<12} | {impact}")

    print("=" * 70)
    print()


if __name__ == "__main__":
    # Run tests
    success = test_decay_curve()
    test_with_base_score()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
