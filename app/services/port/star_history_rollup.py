"""Rollup helpers for project star history retention."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable


@dataclass(frozen=True, slots=True)
class StarPoint:
    """A dated cumulative star snapshot."""

    date: date
    stars: int


def _ensure_monotonic(points: list[StarPoint]) -> list[StarPoint]:
    monotonic: list[StarPoint] = []
    max_stars = 0
    for point in sorted(points, key=lambda item: item.date):
        max_stars = max(max_stars, point.stars)
        monotonic.append(StarPoint(date=point.date, stars=max_stars))
    return monotonic


def rollup_star_points(
    points: Iterable[StarPoint],
    *,
    today: date | None = None,
    recent_days: int = 90,
) -> list[StarPoint]:
    """Keep daily snapshots for recent window and monthly snapshots for older points."""

    source_points = list(points)
    if not source_points:
        return []

    if today is None:
        today = date.today()
    daily_cutoff = today - timedelta(days=recent_days)

    normalized = _ensure_monotonic(source_points)
    daily = [point for point in normalized if point.date >= daily_cutoff]
    older = [point for point in normalized if point.date < daily_cutoff]

    monthly_latest: dict[tuple[int, int], StarPoint] = {}
    for point in older:
        monthly_latest[(point.date.year, point.date.month)] = point

    rolled = list(monthly_latest.values()) + daily
    return sorted(_ensure_monotonic(rolled), key=lambda item: item.date)
