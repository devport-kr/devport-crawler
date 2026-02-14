from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.crawlers.port.projects_stage import ProjectsStage


@dataclass
class FakeProjectRow:
    external_id: str
    port_id: int
    name: str
    full_name: str
    repo_url: str
    homepage_url: str | None
    description: str | None
    stars: int
    forks: int
    contributors: int
    language: str | None
    language_color: str | None
    license: str | None
    tags: list[str] | None
    stars_week_delta: int
    releases_30d: int
    last_release: Any
    created_at: datetime
    updated_at: datetime


class FakeQuery:
    def __init__(self, session: "FakeDB") -> None:
        self._session = session
        self._external_id: str | None = None

    def filter_by(self, **kwargs: Any) -> "FakeQuery":
        self._external_id = kwargs.get("external_id")
        return self

    def first(self) -> FakeProjectRow | None:
        if self._external_id is None:
            return None
        return self._session.projects.get(self._external_id)


class FakeDB:
    def __init__(self) -> None:
        self.projects: dict[str, FakeProjectRow] = {}

    def query(self, _: Any) -> FakeQuery:
        return FakeQuery(self)

    def add(self, row: Any) -> None:
        record = FakeProjectRow(
            external_id=row.external_id,
            port_id=row.port_id,
            name=row.name,
            full_name=row.full_name,
            repo_url=row.repo_url,
            homepage_url=row.homepage_url,
            description=row.description,
            stars=row.stars,
            forks=row.forks,
            contributors=row.contributors,
            language=row.language,
            language_color=row.language_color,
            license=row.license,
            tags=row.tags,
            stars_week_delta=row.stars_week_delta,
            releases_30d=row.releases_30d,
            last_release=row.last_release,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        self.projects[record.external_id] = record


def test_projects_stage_replay_does_not_increase_row_count() -> None:
    stage = ProjectsStage()
    db = FakeDB()
    payload = {
        "id": 101,
        "name": "sdk",
        "full_name": "acme/sdk",
        "html_url": "https://github.com/acme/sdk",
        "homepage": "https://acme.dev/sdk",
        "description": "SDK project",
        "stargazers_count": 120,
        "forks_count": 12,
        "contributors_count": 6,
        "language": "Python",
        "language_color": "#3572A5",
        "topics": ["sdk", "port"],
        "license": {"spdx_id": "MIT"},
    }

    first = stage.ingest_repositories(db, port_id=9, repositories=[payload])
    second = stage.ingest_repositories(db, port_id=9, repositories=[payload])

    assert first["created"] == 1
    assert second["created"] == 0
    assert second["updated"] == 1
    assert len(db.projects) == 1


def test_projects_stage_preserves_existing_optional_values_on_partial_payload() -> None:
    stage = ProjectsStage()
    db = FakeDB()
    full_payload = {
        "id": 202,
        "name": "cli",
        "full_name": "org/cli",
        "html_url": "https://github.com/org/cli",
        "homepage": "https://org.dev/cli",
        "description": "Command line",
        "stargazers_count": 300,
        "forks_count": 25,
        "language": "Go",
        "topics": ["cli", "tooling"],
        "license": {"spdx_id": "Apache-2.0"},
    }
    partial_payload = {
        "id": 202,
        "name": "cli",
        "full_name": "org/cli",
        "html_url": "https://github.com/org/cli",
        "stargazers_count": 310,
        "forks_count": 28,
    }

    stage.ingest_repositories(db, port_id=4, repositories=[full_payload])
    stage.ingest_repositories(db, port_id=4, repositories=[partial_payload])

    row = db.projects["github:202"]
    assert row.stars == 310
    assert row.forks == 28
    assert row.homepage_url == "https://org.dev/cli"
    assert row.language == "Go"
    assert row.license == "Apache-2.0"
    assert row.tags == ["cli", "tooling"]
