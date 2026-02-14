from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.crawlers.port.contracts import FetchResult, FetchState
from app.crawlers.port.star_history_stage import StarHistoryCheckpoint, StarHistoryStage
from app.services.port.star_history_rollup import StarPoint, rollup_star_points


class FakeQuery:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def filter(self, *_: Any, **__: Any) -> "FakeQuery":
        return self

    def all(self) -> list[Any]:
        return self._rows


class FakeDB:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self.rows = rows or []

    def query(self, _: Any) -> FakeQuery:
        return FakeQuery(self.rows)

    def add(self, row: Any) -> None:
        self.rows.append(row)


@dataclass
class FakeProject:
    id: int
    full_name: str


class FakeClient:
    def __init__(self, pages: dict[int, FetchResult[list[dict[str, Any]]]]) -> None:
        self.pages = pages

    async def list_stargazers(self, _: str, __: str, *, page: int, per_page: int = 100) -> FetchResult[list[dict[str, Any]]]:
        return self.pages.get(page, FetchResult(state=FetchState.EMPTY, data=[]))


def test_rollup_keeps_recent_daily_and_rolls_older_monthly() -> None:
    today = date(2026, 2, 14)
    points = [StarPoint(date=today - timedelta(days=offset), stars=100 + offset) for offset in range(0, 140)]

    rolled = rollup_star_points(points, today=today, recent_days=90)

    recent_count = sum(1 for point in rolled if point.date >= today - timedelta(days=90))
    older_count = len(rolled) - recent_count

    assert recent_count == 91
    assert 1 <= older_count <= 3
    assert all(rolled[index].stars <= rolled[index + 1].stars for index in range(len(rolled) - 1))


@pytest.mark.asyncio
async def test_star_history_stage_supports_resume_checkpoint_and_caps() -> None:
    first_page = [
        {"starred_at": "2026-02-14T00:00:00Z"},
        {"starred_at": "2026-02-13T00:00:00Z"},
    ]
    second_page = [
        {"starred_at": "2026-02-12T00:00:00Z"},
        {"starred_at": "2026-02-11T00:00:00Z"},
    ]
    client = FakeClient(
        {
            2: FetchResult(state=FetchState.OK, data=second_page),
            3: FetchResult(state=FetchState.OK, data=first_page),
        }
    )
    db = FakeDB()
    stage = StarHistoryStage(client, max_pages=1, recent_days=90)

    result = await stage.ingest_project(
        db,
        FakeProject(id=10, full_name="owner/repo"),
        checkpoint=StarHistoryCheckpoint(next_page=2),
    )

    assert result.fetched_pages == 1
    assert result.checkpoint.next_page == 3
    assert result.checkpoint.reached_cap is True
    assert result.stored_points >= 1


@pytest.mark.asyncio
async def test_star_history_upsert_is_replay_safe_and_monotonic() -> None:
    db = FakeDB(
        rows=[
            SimpleNamespace(project_id=42, date=date(2026, 2, 10), stars=5),
            SimpleNamespace(project_id=42, date=date(2026, 2, 11), stars=7),
        ]
    )
    client = FakeClient(
        {
            1: FetchResult(
                state=FetchState.OK,
                data=[
                    {"starred_at": "2026-02-10T00:00:00Z"},
                    {"starred_at": "2026-02-10T01:00:00Z"},
                    {"starred_at": "2026-02-11T00:00:00Z"},
                ],
            ),
            2: FetchResult(state=FetchState.EMPTY, data=[]),
        }
    )
    stage = StarHistoryStage(client, max_pages=10, recent_days=90)

    result = await stage.ingest_project(db, FakeProject(id=42, full_name="owner/repo"))

    by_date = {row.date: row.stars for row in db.rows}
    assert result.failed is False
    assert by_date[date(2026, 2, 10)] >= 5
    assert by_date[date(2026, 2, 11)] >= 7
    sorted_days = sorted(by_date)
    assert all(by_date[sorted_days[index]] <= by_date[sorted_days[index + 1]] for index in range(len(sorted_days) - 1))
