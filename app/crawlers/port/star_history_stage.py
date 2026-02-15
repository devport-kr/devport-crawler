"""Project star-history ingestion stage with retention rollups."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from dateutil import parser as date_parser

from app.config.settings import settings
from app.crawlers.port.client import sanitize_log_extra
from app.crawlers.port.contracts import FetchState
from app.models.project import Project
from app.models.project_star_history import ProjectStarHistory
from app.services.port.star_history_rollup import StarPoint, rollup_star_points

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StarHistoryCheckpoint:
    """Resumable cursor for large stargazer backfills."""

    next_page: int = 1
    reached_cap: bool = False
    complete: bool = False


@dataclass(slots=True)
class StarHistoryIngestionResult:
    """Result metadata for one project ingestion run."""

    fetched_pages: int
    source_points: int
    stored_points: int
    checkpoint: StarHistoryCheckpoint
    failed: bool = False


class StarHistoryStage:
    """Ingest stargazer history and persist rolled-up snapshots."""

    def __init__(
        self,
        github_client: Any,
        *,
        max_pages: Optional[int] = None,
        per_page: int = 100,
        recent_days: int = 90,
    ) -> None:
        self._github_client = github_client
        self._max_pages = max_pages or settings.PORT_BACKFILL_MAX_STARGAZER_PAGES
        self._per_page = per_page
        self._recent_days = recent_days

    async def ingest_project(
        self,
        db: Any,
        project: Project,
        *,
        checkpoint: Optional[StarHistoryCheckpoint] = None,
    ) -> StarHistoryIngestionResult:
        owner, repo = self._split_repo(project.full_name)
        current_page = checkpoint.next_page if checkpoint else 1
        fetched_pages = 0
        star_dates: list[date] = []
        failed = False
        complete = False

        while fetched_pages < self._max_pages:
            response = await self._github_client.list_stargazers(
                owner,
                repo,
                page=current_page,
                per_page=self._per_page,
            )
            if response.state == FetchState.FAILED:
                failed = True
                logger.warning(
                    "Failed to fetch stargazer page",
                    extra=sanitize_log_extra(
                        project=project.full_name,
                        page=current_page,
                        error=response.error,
                        stage="star_history",
                    ),
                )
                break

            if response.state == FetchState.UNCHANGED and fetched_pages == 0:
                complete = True
                break

            if response.state == FetchState.EMPTY or not response.data:
                complete = True
                break

            fetched_pages += 1
            current_page += 1
            star_dates.extend(self._extract_star_dates(response.data))

        source_points = self._build_cumulative_points(star_dates)
        if not source_points:
            fallback_stars = max(int(getattr(project, "stars", 0) or 0), 0)
            if fallback_stars > 0:
                source_points = [StarPoint(date.today(), fallback_stars)]
        elif int(getattr(project, "stars", 0) or 0) > 0:
            latest_known = max(int(source_points[-1].stars), int(getattr(project, "stars", 0) or 0))
            source_points.append(StarPoint(date.today(), latest_known))

        rolled_points = rollup_star_points(source_points, recent_days=self._recent_days)
        stored_count = self._upsert_points(db, project.id, rolled_points)

        reached_cap = not complete and not failed and fetched_pages >= self._max_pages
        return StarHistoryIngestionResult(
            fetched_pages=fetched_pages,
            source_points=len(source_points),
            stored_points=stored_count,
            checkpoint=StarHistoryCheckpoint(
                next_page=current_page,
                reached_cap=reached_cap,
                complete=complete,
            ),
            failed=failed,
        )

    @staticmethod
    def _split_repo(full_name: str) -> tuple[str, str]:
        owner, repo = full_name.split("/", 1)
        return owner.strip(), repo.strip()

    @staticmethod
    def _extract_star_dates(payload: list[dict[str, Any]]) -> list[date]:
        dates: list[date] = []
        for item in payload:
            raw = item.get("starred_at")
            if not isinstance(raw, str):
                continue
            try:
                dates.append(date_parser.isoparse(raw).date())
            except (TypeError, ValueError):
                continue
        return dates

    @staticmethod
    def _build_cumulative_points(star_dates: list[date]) -> list[StarPoint]:
        if not star_dates:
            return []

        by_date: dict[date, int] = {}
        for day in star_dates:
            by_date[day] = by_date.get(day, 0) + 1

        total = 0
        points: list[StarPoint] = []
        for day in sorted(by_date.keys()):
            total += by_date[day]
            points.append(StarPoint(day, total))
        return points

    @staticmethod
    def _upsert_points(db: Any, project_id: int, points: list[StarPoint]) -> int:
        if not points:
            return 0

        existing_rows = (
            db.query(ProjectStarHistory)
            .filter(ProjectStarHistory.project_id == project_id)
            .all()
        )
        existing_by_date = {row.date: row for row in existing_rows}

        max_seen = 0
        for day in sorted(existing_by_date):
            max_seen = max(max_seen, int(existing_by_date[day].stars))

        stored = 0
        for point in sorted(points, key=lambda item: item.date):
            max_seen = max(max_seen, int(point.stars))
            row = existing_by_date.get(point.date)
            if row is None:
                row = ProjectStarHistory(project_id=project_id, date=point.date, stars=max_seen)
                db.add(row)
                existing_by_date[point.date] = row
            else:
                row.stars = max(int(row.stars), max_seen)
            stored += 1

        return stored
