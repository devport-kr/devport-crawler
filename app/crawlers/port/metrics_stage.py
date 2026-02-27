"""Daily metrics ingestion stage with replay-safe writes and project rollups."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Sequence

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config.settings import settings
from app.crawlers.port.client import sanitize_log_extra
from app.models.project import Project
from app.models.project_metrics_daily import ProjectMetricsDaily
from app.services.port.project_mapper import map_metrics_to_daily_row

logger = logging.getLogger(__name__)


class MetricsStage:
    """Ingests `project_metrics_daily` snapshots and updates project-level rollups."""

    def ingest_daily_metrics(
        self,
        db: Any,
        *,
        metrics_payloads: Sequence[dict[str, Any]],
        snapshot_date: date,
    ) -> dict[str, int]:
        created = 0
        updated = 0
        failed = 0

        rows: list[dict[str, Any]] = []
        for payload in metrics_payloads:
            try:
                project_id = int(payload["project_id"])
                existing = db.query(ProjectMetricsDaily).filter_by(project_id=project_id, date=snapshot_date).first()
                row = map_metrics_to_daily_row(
                    project_id=project_id,
                    snapshot_date=snapshot_date,
                    metrics_payload=payload,
                    existing=existing,
                )
                rows.append(row)

                if existing is None:
                    created += 1
                else:
                    updated += 1
            except Exception as exc:
                failed += 1
                logger.warning(
                    "Skipping metrics payload due to ingestion error",
                    extra=sanitize_log_extra(error=str(exc), stage="metrics", payload=payload),
                )

        self._upsert_metrics_rows(db, rows)

        for row in rows:
            self._sync_project_rollup(db, row)

        return {
            "input": len(metrics_payloads),
            "created": created,
            "updated": updated,
            "failed": failed,
            "processed": created + updated,
        }

    @staticmethod
    def capped_backfill_days(requested_days: int) -> int:
        """Honor full-history intent while applying practical configured cap."""
        cap = max(int(getattr(settings, "PORT_METRICS_HISTORY_DAYS_CAP", 730)), 1)
        return max(1, min(requested_days, cap))

    def backfill_dates(self, *, end_date: date, requested_days: int) -> list[date]:
        days = self.capped_backfill_days(requested_days)
        return [end_date - timedelta(days=index) for index in range(days)]

    @staticmethod
    def _upsert_metrics_rows(db: Any, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return

        statement = pg_insert(ProjectMetricsDaily).values(
            [
                {
                    "project_id": row["project_id"],
                    "date": row["date"],
                    "stars": row["stars"],
                    "forks": row["forks"],
                    "open_issues": row["open_issues"],
                    "contributors": row["contributors"],
                }
                for row in rows
            ]
        )
        statement = statement.on_conflict_do_update(
            index_elements=[ProjectMetricsDaily.project_id, ProjectMetricsDaily.date],
            set_={
                "stars": statement.excluded.stars,
                "forks": statement.excluded.forks,
                "open_issues": statement.excluded.open_issues,
                "contributors": statement.excluded.contributors,
            },
        )

        if hasattr(db, "execute"):
            try:
                db.execute(statement)
                return
            except Exception:
                # Non-PostgreSQL sessions/tests fall through to portable upsert.
                pass

        for row in rows:
            existing = db.query(ProjectMetricsDaily).filter_by(project_id=row["project_id"], date=row["date"]).first()
            if existing is None:
                db.add(
                    ProjectMetricsDaily(
                        project_id=row["project_id"],
                        date=row["date"],
                        stars=row["stars"],
                        forks=row["forks"],
                        open_issues=row["open_issues"],
                        contributors=row["contributors"],
                    )
                )
                continue

            existing.stars = row["stars"]
            existing.forks = row["forks"]
            existing.open_issues = row["open_issues"]
            existing.contributors = row["contributors"]

    @staticmethod
    def _sync_project_rollup(db: Any, row: dict[str, Any]) -> None:
        project = db.query(Project).filter_by(id=row["project_id"]).first()
        if project is None:
            return

        project.stars = row["stars"]
        project.forks = row["forks"]
        project.contributors = row["contributors"]
