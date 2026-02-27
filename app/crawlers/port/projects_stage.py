"""Projects dataset ingestion stage with replay-safe upsert behavior."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from app.models.project import Project
from app.crawlers.port.client import sanitize_log_extra
from app.services.port.project_mapper import build_project_external_id, map_repo_to_project_row

logger = logging.getLogger(__name__)


class ProjectsStage:
    """Ingests GitHub repositories into the `projects` dataset."""

    def ingest_repositories(self, db: Any, *, repositories: Sequence[dict[str, Any]]) -> dict[str, int]:
        """Upsert project rows by stable external identity without destructive clearing."""

        created = 0
        updated = 0
        failed = 0

        for repo_payload in repositories:
            try:
                external_id = build_project_external_id(repo_payload)
                existing = db.query(Project).filter_by(external_id=external_id).first()
                row = map_repo_to_project_row(repo_payload=repo_payload, existing=existing)

                if existing is None:
                    db.add(Project(**row))
                    created += 1
                else:
                    for field_name, value in row.items():
                        setattr(existing, field_name, value)
                    updated += 1
            except Exception as exc:
                failed += 1
                logger.warning(
                    "Skipping repository due to ingestion error",
                    extra=sanitize_log_extra(
                        error=str(exc),
                        external_id=repo_payload.get("full_name") or repo_payload.get("id") or "unknown",
                        stage="projects",
                    ),
                )

        return {
            "input": len(repositories),
            "created": created,
            "updated": updated,
            "failed": failed,
            "processed": created + updated,
        }
