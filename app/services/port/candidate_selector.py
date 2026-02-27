"""Hybrid seed + auto-candidate selector for Port project ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import log1p
from typing import Iterable, Sequence

from app.config.settings import settings



# Candidate score defaults (relevance gate + weighted stars/activity ranking).
DEFAULT_WEIGHT_RELEVANCE = 0.55
DEFAULT_WEIGHT_STARS = 0.30
DEFAULT_WEIGHT_ACTIVITY = 0.15
DEFAULT_DIVERSITY_PENALTY = 0.12

# Filtering thresholds for AI-focused / trending repos.
DEFAULT_MAX_INACTIVE_DAYS = 180


@dataclass(slots=True)
class RepoCandidate:
    """Minimal repository representation used by selector logic."""

    external_id: str
    full_name: str
    description: str = ""
    topics: tuple[str, ...] = ()
    stars: int = 0
    pushed_at: datetime | None = None
    archived: bool = False
    disabled: bool = False

    @property
    def owner(self) -> str:
        return self.full_name.split("/", 1)[0].lower() if "/" in self.full_name else "unknown"


@dataclass(slots=True)
class SelectionConfig:
    """Configurable scoring and scope knobs."""

    global_target: int = field(
        default_factory=lambda: getattr(settings, "PORT_PROJECT_GLOBAL_TARGET", 1000)
    )
    relevance_weight: float = field(
        default_factory=lambda: getattr(settings, "PORT_CANDIDATE_WEIGHT_RELEVANCE", DEFAULT_WEIGHT_RELEVANCE)
    )
    stars_weight: float = field(
        default_factory=lambda: getattr(settings, "PORT_CANDIDATE_WEIGHT_STARS", DEFAULT_WEIGHT_STARS)
    )
    activity_weight: float = field(
        default_factory=lambda: getattr(settings, "PORT_CANDIDATE_WEIGHT_ACTIVITY", DEFAULT_WEIGHT_ACTIVITY)
    )
    diversity_soft_cap: int = field(
        default_factory=lambda: getattr(settings, "PORT_CANDIDATE_DIVERSITY_SOFT_CAP", 3)
    )
    diversity_penalty: float = DEFAULT_DIVERSITY_PENALTY
    max_inactive_days: int = field(
        default_factory=lambda: getattr(settings, "PORT_CANDIDATE_MAX_INACTIVE_DAYS", DEFAULT_MAX_INACTIVE_DAYS)
    )
    now_provider: callable = datetime.now


class CandidateSelector:
    """Selects global repositories from manual baseline and automatic expansion."""

    def __init__(self, config: SelectionConfig | None = None) -> None:
        self.config = config or SelectionConfig()

    def select_candidates(
        self,
        *,
        manual_baseline: Sequence[RepoCandidate],
        auto_candidates: Sequence[RepoCandidate],
        relevance_keywords: Iterable[str],
        target_count: int | None = None,
    ) -> list[RepoCandidate]:
        """Merge baseline + auto pool with relevance gate and soft diversity."""

        target = self._clamp_target(target_count)
        keywords = {k.lower() for k in relevance_keywords if k and k.strip()}

        selected: list[RepoCandidate] = []
        selected_ids: set[str] = set()
        org_counts: dict[str, int] = {}

        for candidate in manual_baseline:
            if candidate.external_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.external_id)
            org_counts[candidate.owner] = org_counts.get(candidate.owner, 0) + 1
            if len(selected) >= target:
                return selected

        filtered_auto = [
            candidate
            for candidate in auto_candidates
            if candidate.external_id not in selected_ids and self._passes_relevance_gate(candidate, keywords)
        ]

        while filtered_auto and len(selected) < target:
            scored = [
                (self._weighted_score(candidate, keywords) - self._diversity_penalty(candidate, org_counts), candidate)
                for candidate in filtered_auto
            ]
            scored.sort(key=lambda item: (-item[0], item[1].full_name.lower()))
            chosen = scored[0][1]
            selected.append(chosen)
            selected_ids.add(chosen.external_id)
            org_counts[chosen.owner] = org_counts.get(chosen.owner, 0) + 1
            filtered_auto = [c for c in filtered_auto if c.external_id != chosen.external_id]

        return selected

    def _clamp_target(self, target_count: int | None) -> int:
        return max(1, target_count if target_count is not None else self.config.global_target)

    def _passes_relevance_gate(self, candidate: RepoCandidate, keywords: set[str]) -> bool:
        if candidate.archived or candidate.disabled:
            return False

        # Recency gate: exclude repos not pushed in the last N days
        if candidate.pushed_at is not None:
            now = self.config.now_provider(UTC)
            pushed = candidate.pushed_at if candidate.pushed_at.tzinfo else candidate.pushed_at.replace(tzinfo=UTC)
            cutoff = now - timedelta(days=self.config.max_inactive_days)
            if pushed < cutoff:
                return False

        if not keywords:
            return True

        haystack = " ".join([candidate.full_name, candidate.description, *candidate.topics]).lower()
        return any(keyword in haystack for keyword in keywords)

    def _weighted_score(self, candidate: RepoCandidate, keywords: set[str]) -> float:
        relevance = self._relevance_score(candidate, keywords)
        stars = self._stars_score(candidate)
        activity = self._activity_score(candidate)
        return (
            relevance * self.config.relevance_weight
            + stars * self.config.stars_weight
            + activity * self.config.activity_weight
        )

    def _relevance_score(self, candidate: RepoCandidate, keywords: set[str]) -> float:
        if not keywords:
            return 1.0
        haystack = " ".join([candidate.full_name, candidate.description, *candidate.topics]).lower()
        matches = sum(1 for keyword in keywords if keyword in haystack)
        return min(matches / max(len(keywords), 1), 1.0)

    @staticmethod
    def _stars_score(candidate: RepoCandidate) -> float:
        return min(log1p(max(candidate.stars, 0)) / log1p(50000), 1.0)

    def _activity_score(self, candidate: RepoCandidate) -> float:
        if candidate.pushed_at is None:
            return 0.0
        now = self.config.now_provider(UTC)
        pushed = candidate.pushed_at if candidate.pushed_at.tzinfo else candidate.pushed_at.replace(tzinfo=UTC)
        days = max((now - pushed).days, 0)
        return max(0.0, 1 - (days / 365))

    def _diversity_penalty(self, candidate: RepoCandidate, org_counts: dict[str, int]) -> float:
        current = org_counts.get(candidate.owner, 0)
        excess = max(0, current - self.config.diversity_soft_cap + 1)
        return float(excess) * self.config.diversity_penalty
