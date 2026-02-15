"""Project events ingestion stage with strict fallback ordering."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from dateutil import parser as date_parser

from app.crawlers.port.client import sanitize_log_extra, sanitize_for_log
from app.crawlers.port.contracts import FetchResult, FetchState
from app.models.project import Project
from app.models.project_event import ProjectEvent
from app.services.port.event_classifier import classify_event

logger = logging.getLogger(__name__)


CHANGELOG_PATH_CANDIDATES = (
    "CHANGELOG.md",
    "CHANGELOG",
    "CHANGELOG.txt",
    "changelog.md",
    "docs/changelog.md",
    "docs/changelog",
    "docs/changelog.txt",
    "releases.md",
)

_TYPE_SUMMARY_KO = {
    "security": "보안 패치",
    "breaking": "호환성 영향 변경",
    "feature": "새 기능",
    "fix": "버그 수정",
    "perf": "성능 개선",
    "misc": "유지보수 업데이트",
}


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


class EventsStage:
    """Ingests `project_events` using release/tag/changelog fallback."""

    def __init__(self, github_client: Any) -> None:
        self._github_client = github_client

    async def ingest_project(self, db: Any, project: Project) -> EventIngestionResult:
        owner, repo = self._split_repo(project.full_name)
        failure_reasons: list[str] = []

        releases = await self._github_client.list_releases(owner, repo)
        if releases.is_ok and releases.data:
            rows = self._normalize_release_rows(project.id, releases.data)
            self._upsert_events(db, project.id, rows)
            return EventIngestionResult("releases", len(rows), False, failure_reasons)
        if releases.is_failed:
            failure_reasons.append(sanitize_for_log(f"releases: {releases.error or 'unknown'}", key="error"))

        tags = await self._github_client.list_tags(owner, repo)
        if tags.is_ok and tags.data:
            rows = self._normalize_tag_rows(project.id, tags.data)
            self._upsert_events(db, project.id, rows)
            return EventIngestionResult("tags", len(rows), False, failure_reasons)
        if tags.is_failed:
            failure_reasons.append(sanitize_for_log(f"tags: {tags.error or 'unknown'}", key="error"))

        changelog, changelog_failures = await self._fetch_changelog(owner, repo)
        failure_reasons.extend(changelog_failures)
        if changelog is not None:
            rows = self._normalize_changelog_rows(project.id, changelog)
            self._upsert_events(db, project.id, rows)
            return EventIngestionResult("changelog", len(rows), False, failure_reasons)

        if releases.is_failed and tags.is_failed and changelog_failures:
            logger.warning(
                "Skipping project event update because all sources failed",
                extra=sanitize_log_extra(project=project.full_name, reasons=failure_reasons, stage="events"),
            )
            return EventIngestionResult(None, 0, True, failure_reasons)

        return EventIngestionResult(None, 0, False, failure_reasons)

    @staticmethod
    def _split_repo(full_name: str) -> tuple[str, str]:
        owner, repo = full_name.split("/", 1)
        return owner.strip(), repo.strip()

    async def _fetch_changelog(self, owner: str, repo: str) -> tuple[Optional[str], list[str]]:
        failures: list[str] = []
        for path in CHANGELOG_PATH_CANDIDATES:
            response: FetchResult[str] = await self._github_client.get_content(owner, repo, path)
            if response.state == FetchState.OK and response.data:
                return response.data, failures
            if response.is_failed and response.status_code not in (404,):
                failures.append(sanitize_for_log(f"changelog({path}): {response.error or 'unknown'}", key="error"))
        return None, failures

    def _normalize_release_rows(self, project_id: int, releases: list[dict[str, Any]]) -> list[EventRow]:
        rows: list[EventRow] = []
        for release in releases:
            version = str(release.get("tag_name") or release.get("name") or "unknown")
            body = str(release.get("body") or "")
            title = str(release.get("name") or version)
            summary = self._build_korean_summary(version=version, title=title, raw_notes=body)
            released_at = self._parse_date(release.get("published_at") or release.get("created_at"))
            source_url = release.get("html_url")
            rows.append(self._build_row(project_id, version, released_at, summary, body, source_url))
        return rows

    def _normalize_tag_rows(self, project_id: int, tags: list[dict[str, Any]]) -> list[EventRow]:
        rows: list[EventRow] = []
        for tag in tags:
            version = str(tag.get("name") or "unknown")
            commit = tag.get("commit") if isinstance(tag.get("commit"), dict) else {}
            sha = str(commit.get("sha") or "")
            source_url = commit.get("url") if isinstance(commit.get("url"), str) else None
            summary = f"{version} 태그 릴리스: 버전 식별용 스냅샷이 반영되었습니다."
            body = sha
            released_at = date.today()
            rows.append(self._build_row(project_id, version, released_at, summary, body, source_url))
        return rows

    def _normalize_changelog_rows(self, project_id: int, changelog_text: str) -> list[EventRow]:
        rows: list[EventRow] = []
        matches = list(re.finditer(r"^##\s*\[?([^\]\n]+)\]?", changelog_text, flags=re.MULTILINE))
        if not matches:
            return [
                self._build_row(
                    project_id,
                    "changelog",
                    date.today(),
                    "Changelog update",
                    changelog_text,
                    None,
                )
            ]

        for index, match in enumerate(matches):
            version = match.group(1).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(changelog_text)
            notes = changelog_text[start:end].strip()
            rows.append(
                self._build_row(
                    project_id,
                    version or f"changelog-{index + 1}",
                    date.today(),
                    self._build_korean_summary(
                        version=version or f"changelog-{index + 1}",
                        title=f"Changelog {version}",
                        raw_notes=notes,
                    ),
                    notes,
                    None,
                )
            )
        return rows

    def _build_korean_summary(self, *, version: str, title: str, raw_notes: str) -> str:
        classification = classify_event(title=title, body=raw_notes)
        labels: list[str] = []
        for event_type in classification.event_types:
            labels.append(_TYPE_SUMMARY_KO.get(event_type) or _TYPE_SUMMARY_KO["misc"])
        unique_labels = list(dict.fromkeys(labels))
        labels_text = ", ".join(unique_labels[:3]) if unique_labels else _TYPE_SUMMARY_KO["misc"]

        lead = f"{version} 릴리스: {labels_text} 중심 변경이 포함되었습니다."
        note = self._extract_primary_note(raw_notes)
        if note:
            return f"{lead} 주요 변경: {note}"
        return lead

    @staticmethod
    def _extract_primary_note(raw_notes: str) -> str:
        for line in raw_notes.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            if candidate.startswith("#"):
                continue
            candidate = re.sub(r"^[-*+>\s]+", "", candidate).strip()
            if not candidate:
                continue
            candidate = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", candidate)
            candidate = re.sub(r"`([^`]+)`", r"\1", candidate)
            candidate = candidate.replace("\t", " ").strip()
            if candidate:
                return candidate[:320]
        return ""

    def _build_row(
        self,
        project_id: int,
        version: str,
        released_at: date,
        summary: str,
        raw_notes: str,
        source_url: Optional[str],
    ) -> EventRow:
        classification = classify_event(title=summary, body=raw_notes)
        digest = hashlib.sha1(f"{project_id}:{version}:{released_at.isoformat()}".encode("utf-8")).hexdigest()[:16]
        bullets = [line.strip("- ").strip() for line in raw_notes.splitlines() if line.strip().startswith("-")][:8]
        return EventRow(
            external_id=f"evt-{project_id}-{digest}",
            version=version[:50],
            released_at=released_at,
            summary=summary[:1000],
            raw_notes=raw_notes,
            source_url=source_url,
            event_types=classification.event_types,
            impact_score=classification.impact_score,
            is_security=classification.is_security,
            is_breaking=classification.is_breaking,
            bullets=bullets,
        )

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
