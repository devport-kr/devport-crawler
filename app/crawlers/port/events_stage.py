"""Project events ingestion stage — releases/tags only, no changelog scanning."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

from dateutil import parser as date_parser

from app.crawlers.port.client import sanitize_log_extra, sanitize_for_log
from app.crawlers.port.contracts import FetchResult, FetchState
from app.models.project import Project
from app.models.project_event import ProjectEvent

logger = logging.getLogger(__name__)

# Simple keyword-based type labels (English, no LLM)
_KEYWORD_TYPES: list[tuple[str, list[str]]] = [
    ("security", ["security", "cve", "vulnerability", "exploit", "auth", "injection"]),
    ("breaking", ["breaking", "incompatible", "removed", "deprecated", "migration"]),
    ("feature", ["feat", "feature", "add", "new", "introduce", "support"]),
    ("fix", ["fix", "bug", "patch", "correct", "resolve", "issue", "error"]),
    ("perf", ["perf", "performance", "speed", "optim", "faster", "latency"]),
]


@dataclass(slots=True)
class EventRow:
    """Persistence-oriented representation of a project event."""

    external_id: str
    version: str
    released_at: date
    summary: str
    raw_notes: str
    source_url: Optional[str]
    event_types: list[str]
    impact_score: int
    is_security: bool
    is_breaking: bool
    bullets: list[str]


@dataclass(slots=True)
class EventIngestionResult:
    """Execution result for a single project ingestion run."""

    source: Optional[str]
    updated_count: int
    skipped_event_update: bool
    failure_reasons: list[str]
    last_release_date: Optional[date]


class EventsStage:
    """Ingests `project_events` using releases API with tags fallback."""

    def __init__(self, github_client: Any) -> None:
        self._github_client = github_client

    async def ingest_project(self, db: Any, project: Project) -> EventIngestionResult:
        owner, repo = self._split_repo(project.full_name)
        failure_reasons: list[str] = []

        releases = await self._github_client.list_releases(owner, repo)
        if releases.is_ok and releases.data:
            rows = self._normalize_release_rows(project.id, releases.data)
            self._upsert_events(db, project.id, rows)
            last_release = max((r.released_at for r in rows), default=None)
            return EventIngestionResult("releases", len(rows), False, failure_reasons, last_release)
        if releases.is_failed:
            failure_reasons.append(sanitize_for_log(f"releases: {releases.error or 'unknown'}", key="error"))

        tags = await self._github_client.list_tags(owner, repo)
        if tags.is_ok and tags.data:
            rows = self._normalize_tag_rows(project.id, tags.data)
            self._upsert_events(db, project.id, rows)
            last_release = max((r.released_at for r in rows), default=None)
            return EventIngestionResult("tags", len(rows), False, failure_reasons, last_release)
        if tags.is_failed:
            failure_reasons.append(sanitize_for_log(f"tags: {tags.error or 'unknown'}", key="error"))

        logger.warning(
            "Skipping project event update because all sources failed",
            extra=sanitize_log_extra(project=project.full_name, reasons=failure_reasons, stage="events"),
        )
        return EventIngestionResult(None, 0, True, failure_reasons, None)

    @staticmethod
    def _split_repo(full_name: str) -> tuple[str, str]:
        owner, repo = full_name.split("/", 1)
        return owner.strip(), repo.strip()

    def _normalize_release_rows(self, project_id: int, releases: list[dict[str, Any]]) -> list[EventRow]:
        rows: list[EventRow] = []
        for release in releases:
            version = str(release.get("tag_name") or release.get("name") or "unknown")
            body = str(release.get("body") or "")
            title = str(release.get("name") or version)
            released_at = self._parse_date(release.get("published_at") or release.get("created_at"))
            source_url = release.get("html_url")
            rows.append(self._build_row(project_id, version, title, released_at, body, source_url))
        return rows

    def _normalize_tag_rows(self, project_id: int, tags: list[dict[str, Any]]) -> list[EventRow]:
        rows: list[EventRow] = []
        for tag in tags:
            version = str(tag.get("name") or "unknown")
            commit = tag.get("commit") if isinstance(tag.get("commit"), dict) else {}
            sha = str(commit.get("sha") or "")
            source_url = commit.get("url") if isinstance(commit.get("url"), str) else None
            released_at = date.today()
            rows.append(self._build_row(project_id, version, version, released_at, sha, source_url))
        return rows

    def _build_row(
        self,
        project_id: int,
        version: str,
        title: str,
        released_at: date,
        raw_notes: str,
        source_url: Optional[str],
    ) -> EventRow:
        event_types = self._classify(title, raw_notes)
        is_security = "security" in event_types
        is_breaking = "breaking" in event_types
        impact_score = self._compute_impact(event_types)
        summary = self._build_summary(version, event_types)
        bullets = [
            line.strip("- ").strip()
            for line in raw_notes.splitlines()
            if line.strip().startswith("-")
        ][:5]
        digest = hashlib.sha1(f"{project_id}:{version}:{released_at.isoformat()}".encode()).hexdigest()[:16]
        return EventRow(
            external_id=f"evt-{project_id}-{digest}",
            version=version[:50],
            released_at=released_at,
            summary=summary[:1000],
            raw_notes=raw_notes,
            source_url=source_url,
            event_types=event_types,
            impact_score=impact_score,
            is_security=is_security,
            is_breaking=is_breaking,
            bullets=bullets,
        )

    @staticmethod
    def _classify(title: str, body: str) -> list[str]:
        text = f"{title} {body}".lower()
        found: list[str] = []
        for label, keywords in _KEYWORD_TYPES:
            if any(kw in text for kw in keywords):
                found.append(label)
        return found or ["misc"]

    @staticmethod
    def _compute_impact(event_types: list[str]) -> int:
        score = 1
        if "security" in event_types:
            score += 4
        if "breaking" in event_types:
            score += 3
        if "feature" in event_types:
            score += 2
        if "fix" in event_types:
            score += 1
        if "perf" in event_types:
            score += 1
        return min(score, 10)

    @staticmethod
    def _build_summary(version: str, event_types: list[str]) -> str:
        labels = ", ".join(event_types[:3])
        return f"{version}: {labels}"

    @staticmethod
    def _parse_date(raw: Any) -> date:
        if isinstance(raw, str) and raw.strip():
            try:
                return date_parser.isoparse(raw).date()
            except (TypeError, ValueError):
                pass
        return date.today()

    @staticmethod
    def _upsert_events(db: Any, project_id: int, rows: list[EventRow]) -> None:
        if not rows:
            return

        keys = [row.external_id for row in rows]
        existing = {
            event.external_id: event
            for event in db.query(ProjectEvent).filter(ProjectEvent.external_id.in_(keys)).all()
        }

        for row in rows:
            event = existing.get(row.external_id)
            if event is None:
                event = ProjectEvent(external_id=row.external_id, project_id=project_id)
                db.add(event)

            event.version = row.version
            event.released_at = row.released_at
            event.summary = row.summary
            event.raw_notes = row.raw_notes
            event.source_url = row.source_url
            event.event_types = row.event_types
            event.impact_score = row.impact_score
            event.is_security = row.is_security
            event.is_breaking = row.is_breaking
            event.bullets = row.bullets
