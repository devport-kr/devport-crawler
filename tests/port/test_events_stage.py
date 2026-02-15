from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from app.crawlers.port.contracts import FetchResult, FetchState
from app.crawlers.port.events_stage import EventsStage


@dataclass
class FakeProject:
    id: int
    full_name: str


class FakeQuery:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def filter(self, *_: Any, **__: Any) -> "FakeQuery":
        return self

    def all(self) -> list[Any]:
        return self._rows


class FakeDB:
    def __init__(self) -> None:
        self.rows: list[SimpleNamespace] = []

    def query(self, _: Any) -> FakeQuery:
        return FakeQuery(self.rows)

    def add(self, row: Any) -> None:
        self.rows.append(row)


class FakeClient:
    def __init__(self) -> None:
        self.release_result = FetchResult(state=FetchState.EMPTY, data=[])
        self.tag_result = FetchResult(state=FetchState.EMPTY, data=[])
        self.content_results: dict[str, FetchResult[str]] = {}

    async def list_releases(self, *_: Any, **__: Any) -> FetchResult[list[dict[str, Any]]]:
        return self.release_result

    async def list_tags(self, *_: Any, **__: Any) -> FetchResult[list[dict[str, Any]]]:
        return self.tag_result

    async def get_content(self, _: str, __: str, path: str) -> FetchResult[str]:
        return self.content_results.get(path, FetchResult(state=FetchState.FAILED, status_code=404, error="not found"))


@pytest.mark.asyncio
async def test_events_stage_prefers_releases_before_tags_and_changelog() -> None:
    client = FakeClient()
    client.release_result = FetchResult(
        state=FetchState.OK,
        data=[
            {
                "tag_name": "v1.2.3",
                "name": "v1.2.3",
                "body": "- Added feature\n- Security patch",
                "published_at": "2026-02-10T00:00:00Z",
                "html_url": "https://github.com/o/r/releases/tag/v1.2.3",
            }
        ],
    )
    client.tag_result = FetchResult(state=FetchState.OK, data=[{"name": "v9.9.9"}])
    client.content_results["CHANGELOG.md"] = FetchResult(state=FetchState.OK, data="## [v0.1.0]\n- test")

    db = FakeDB()
    stage = EventsStage(client)

    result = await stage.ingest_project(db, FakeProject(id=1, full_name="owner/repo"))

    assert result.source == "releases"
    assert result.updated_count == 1
    assert result.skipped_event_update is False
    assert len(db.rows) == 1
    assert db.rows[0].version == "v1.2.3"
    assert db.rows[0].summary != "v1.2.3"
    assert "릴리스" in db.rows[0].summary
    assert "security" in db.rows[0].event_types


@pytest.mark.asyncio
async def test_events_stage_uses_tags_when_releases_empty() -> None:
    client = FakeClient()
    client.release_result = FetchResult(state=FetchState.EMPTY, data=[])
    client.tag_result = FetchResult(state=FetchState.OK, data=[{"name": "v2.0.0", "commit": {"sha": "abc"}}])

    db = FakeDB()
    stage = EventsStage(client)

    result = await stage.ingest_project(db, FakeProject(id=2, full_name="owner/repo"))

    assert result.source == "tags"
    assert result.updated_count == 1
    assert db.rows[0].version == "v2.0.0"
    assert "태그 릴리스" in db.rows[0].summary


@pytest.mark.asyncio
async def test_events_stage_uses_changelog_when_releases_and_tags_empty() -> None:
    client = FakeClient()
    client.release_result = FetchResult(state=FetchState.EMPTY, data=[])
    client.tag_result = FetchResult(state=FetchState.EMPTY, data=[])
    client.content_results["CHANGELOG.md"] = FetchResult(
        state=FetchState.OK,
        data="## [v1.0.0]\n- Breaking change\n## [v0.9.0]\n- Added support",
    )

    db = FakeDB()
    stage = EventsStage(client)

    result = await stage.ingest_project(db, FakeProject(id=3, full_name="owner/repo"))

    assert result.source == "changelog"
    assert result.updated_count == 2
    assert len(db.rows) == 2


@pytest.mark.asyncio
async def test_events_stage_preserves_existing_rows_on_full_source_failure() -> None:
    client = FakeClient()
    client.release_result = FetchResult(state=FetchState.FAILED, error="release timeout", status_code=503)
    client.tag_result = FetchResult(state=FetchState.FAILED, error="tag timeout", status_code=503)
    client.content_results["CHANGELOG.md"] = FetchResult(state=FetchState.FAILED, error="gateway timeout", status_code=504)

    db = FakeDB()
    db.rows = [SimpleNamespace(external_id="existing")]
    stage = EventsStage(client)

    result = await stage.ingest_project(db, FakeProject(id=4, full_name="owner/repo"))

    assert result.skipped_event_update is True
    assert result.source is None
    assert len(result.failure_reasons) >= 3
    assert len(db.rows) == 1
    assert db.rows[0].external_id == "existing"
