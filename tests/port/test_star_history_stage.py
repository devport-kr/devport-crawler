from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from app.crawlers.port.contracts import FetchResult, FetchState
from app.crawlers.port.star_history_stage import StarHistoryStage


@dataclass
class FakeProject:
    id: int
    full_name: str
    stars: int


class FakeQuery:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def filter(self, *_: Any, **__: Any) -> "FakeQuery":
        return self

    def all(self) -> list[Any]:
        return self._rows


class FakeDB:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def query(self, _: Any) -> FakeQuery:
        return FakeQuery(self.rows)

    def add(self, row: Any) -> None:
        self.rows.append(row)


class FailedClient:
    async def list_stargazers(self, *_: Any, **__: Any) -> FetchResult[list[dict[str, Any]]]:
        return FetchResult(state=FetchState.FAILED, error="rate limited", status_code=429)


class OnePageClient:
    async def list_stargazers(self, *_: Any, **__: Any) -> FetchResult[list[dict[str, Any]]]:
        return FetchResult(
            state=FetchState.OK,
            data=[
                {"starred_at": "2026-01-01T00:00:00Z"},
                {"starred_at": "2026-01-02T00:00:00Z"},
            ],
        )


@pytest.mark.asyncio
async def test_star_history_falls_back_to_project_stars_when_api_fails() -> None:
    stage = StarHistoryStage(FailedClient(), max_pages=1)
    db = FakeDB()

    result = await stage.ingest_project(db, FakeProject(id=1, full_name="owner/repo", stars=1234))

    assert result.stored_points == 1
    assert len(db.rows) == 1
    assert db.rows[0].stars == 1234


@pytest.mark.asyncio
async def test_star_history_uses_stargazer_points_when_available() -> None:
    stage = StarHistoryStage(OnePageClient(), max_pages=1)
    db = FakeDB()

    result = await stage.ingest_project(db, FakeProject(id=2, full_name="owner/repo", stars=50))

    assert result.stored_points >= 1
    assert any(getattr(row, "stars", 0) >= 1 for row in db.rows)
    assert any(getattr(row, "date", None) == date.today() for row in db.rows)
