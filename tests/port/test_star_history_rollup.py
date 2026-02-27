from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest


from app.services.port.star_history_rollup import StarPoint, rollup_star_points



def test_rollup_keeps_recent_daily_and_rolls_older_monthly() -> None:
    today = date(2026, 2, 14)
    points = [StarPoint(date=today - timedelta(days=offset), stars=100 + offset) for offset in range(0, 140)]

    rolled = rollup_star_points(points, today=today, recent_days=90)

    recent_count = sum(1 for point in rolled if point.date >= today - timedelta(days=90))
    older_count = len(rolled) - recent_count

    assert recent_count == 91
    assert 1 <= older_count <= 3
    assert all(rolled[index].stars <= rolled[index + 1].stars for index in range(len(rolled) - 1))

