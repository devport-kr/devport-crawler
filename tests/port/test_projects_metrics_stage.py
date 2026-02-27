from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.crawlers.port.metrics_stage import MetricsStage
from app.crawlers.port.projects_stage import ProjectsStage
from app.models.project import Project
from app.models.project_metrics_daily import ProjectMetricsDaily


@dataclass
class FakeProjectRow:
    id: int | None
    external_id: str
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


@dataclass
class FakeMetricRow:
    project_id: int
    date: date
    stars: int
    forks: int
    open_issues: int
    contributors: int


class FakeQuery:
    def __init__(self, session: "FakeDB", model: Any) -> None:
        self._session = session
        self._model = model
        self._filters: dict[str, Any] = {}

    def filter_by(self, **kwargs: Any) -> "FakeQuery":
        self._filters = kwargs
        return self

    def first(self) -> Any:
        if self._model is Project:
            external_id = self._filters.get("external_id")
            project_id = self._filters.get("id")
            if external_id is not None:
                return self._session.projects_by_external_id.get(external_id)
            if project_id is not None:
                return self._session.projects_by_id.get(project_id)
            return None
        if self._model is ProjectMetricsDaily:
            key = (self._filters.get("project_id"), self._filters.get("date"))
            return self._session.metrics.get(key)
        return None


class FakeDB:
    def __init__(self) -> None:
        self.projects_by_external_id: dict[str, FakeProjectRow] = {}
        self.projects_by_id: dict[int, FakeProjectRow] = {}
        self.metrics: dict[tuple[int, date], FakeMetricRow] = {}
        self.next_project_id = 1

    def query(self, model: Any) -> FakeQuery:
        return FakeQuery(self, model)

    def execute(self, _: Any) -> None:
        raise RuntimeError("fake db does not support SQL execution")

    def add(self, row: Any) -> None:
        if hasattr(row, "external_id"):
            record = FakeProjectRow(
                id=self.next_project_id,
                external_id=row.external_id,
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
            self.next_project_id += 1
            self.projects_by_external_id[record.external_id] = record
            self.projects_by_id[record.id] = record
            return

        metric_row = FakeMetricRow(
            project_id=row.project_id,
            date=row.date,
            stars=row.stars,
            forks=row.forks,
            open_issues=row.open_issues,
            contributors=row.contributors,
        )
        self.metrics[(metric_row.project_id, metric_row.date)] = metric_row


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

    first = stage.ingest_repositories(db, repositories=[payload])
    second = stage.ingest_repositories(db, repositories=[payload])

    assert first["created"] == 1
    assert second["created"] == 0
    assert second["updated"] == 1
    assert len(db.projects_by_external_id) == 1


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

    stage.ingest_repositories(db, repositories=[full_payload])
    stage.ingest_repositories(db, repositories=[partial_payload])

    row = db.projects_by_external_id["github:202"]
    assert row.stars == 310
    assert row.forks == 28
    assert row.homepage_url == "https://org.dev/cli"
    assert row.language == "Go"
    assert row.license == "Apache-2.0"
    assert row.tags == ["cli", "tooling"]


def test_metrics_stage_preserves_project_id_date_uniqueness_on_rerun() -> None:
    projects_stage = ProjectsStage()
    metrics_stage = MetricsStage()
    db = FakeDB()
    projects_stage.ingest_repositories(
        db,
        repositories=[
            {
                "id": 909,
                "name": "agent",
                "full_name": "port/agent",
                "html_url": "https://github.com/port/agent",
                "stargazers_count": 400,
                "forks_count": 40,
                "contributors_count": 10,
            }
        ],
    )
    project = db.projects_by_external_id["github:909"]
    snapshot_date = date(2026, 2, 14)
    payload = {
        "project_id": project.id,
        "stargazers_count": 401,
        "forks_count": 41,
        "open_issues_count": 9,
        "contributors_count": 11,
    }

    first = metrics_stage.ingest_daily_metrics(db, metrics_payloads=[payload], snapshot_date=snapshot_date)
    second = metrics_stage.ingest_daily_metrics(db, metrics_payloads=[payload], snapshot_date=snapshot_date)

    assert first["created"] == 1
    assert second["created"] == 0
    assert second["updated"] == 1
    assert len(db.metrics) == 1
    stored = db.metrics[(project.id, snapshot_date)]
    assert stored.stars == 401
    assert stored.open_issues == 9


def test_metrics_stage_updates_project_rollup_fields_from_latest_snapshot() -> None:
    projects_stage = ProjectsStage()
    metrics_stage = MetricsStage()
    db = FakeDB()
    projects_stage.ingest_repositories(
        db,
        repositories=[
            {
                "id": 300,
                "name": "core",
                "full_name": "acme/core",
                "html_url": "https://github.com/acme/core",
                "stargazers_count": 10,
                "forks_count": 1,
                "contributors_count": 1,
            }
        ],
    )
    project = db.projects_by_external_id["github:300"]

    metrics_stage.ingest_daily_metrics(
        db,
        snapshot_date=date(2026, 2, 14),
        metrics_payloads=[
            {
                "project_id": project.id,
                "stargazers_count": 250,
                "forks_count": 30,
                "open_issues_count": 5,
                "contributors_count": 8,
            }
        ],
    )

    updated_project = db.projects_by_id[project.id]
    assert updated_project.stars == 250
    assert updated_project.forks == 30
    assert updated_project.contributors == 8
