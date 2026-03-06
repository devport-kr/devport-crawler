"""Port-domain orchestrator with dataset-isolated stage execution."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
import hashlib
import hmac
import json
import logging
import random
from typing import Any, Callable, Sequence

import httpx

from app.config.database import SessionLocal
from app.config.settings import settings
from app.crawlers.port.client import GitHubPortClient, sanitize_for_log, sanitize_log_extra
from app.crawlers.port.events_stage import EventsStage
from app.crawlers.port.metrics_stage import MetricsStage
from app.models.project import Project

logger = logging.getLogger(__name__)

STAGE_EVENTS = "events"
STAGE_METRICS = "metrics"

DAILY_DEFAULT_STAGES = (STAGE_EVENTS, STAGE_METRICS)
WEBHOOK_SCOPE = "GIT_REPO"


class PortCrawlerOrchestrator:
    """Coordinates isolated execution of port-domain ingestion stages."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = SessionLocal,
        github_client_factory: Callable[[], Any] = GitHubPortClient,
        events_stage: Any | None = None,
        metrics_stage: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._github_client_factory = github_client_factory
        self._events_stage = events_stage
        self._metrics_stage = metrics_stage

    async def run_daily_sync(
        self,
        *,
        stages: Sequence[str] | None = None,
        project_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        selected_stages = tuple(stages or DAILY_DEFAULT_STAGES)
        return await self._run(mode="daily", stages=selected_stages, project_ids=project_ids)

    async def _run(
        self,
        *,
        mode: str,
        stages: Sequence[str],
        project_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        self._ensure_runtime_schema_compatibility()

        started_at = datetime.utcnow().isoformat()
        run_stats: dict[str, Any] = {
            "mode": mode,
            "started_at": started_at,
            "stages_requested": list(stages),
            "stages": {},
            "errors": [],
        }
        logger.info(
            "Port crawler run started",
            extra=sanitize_log_extra(mode=mode, stages_requested=list(stages), project_ids=list(project_ids or [])),
        )

        for stage_name in stages:
            stage_runner = self._resolve_stage_runner(stage_name)
            if stage_runner is None:
                unknown_error = f"Unknown stage: {stage_name}"
                run_stats["stages"][stage_name] = {
                    "success": False,
                    "error": unknown_error,
                    "stats": {},
                }
                run_stats["errors"].append(sanitize_for_log(f"{stage_name}: unknown stage"))
                logger.warning(
                    "Port crawler received unknown stage",
                    extra=sanitize_log_extra(stage=stage_name, error=unknown_error),
                )
                continue

            try:
                result = await stage_runner(
                    project_ids=project_ids,
                )
                stage_error = sanitize_for_log(result.get("error")) if isinstance(result, dict) else None
                if stage_error is not None and isinstance(result, dict):
                    result["error"] = stage_error
                run_stats["stages"][stage_name] = result
                if not result.get("success", False):
                    summarized_error = sanitize_for_log(result.get("error", "stage failed"))
                    run_stats["errors"].append(f"{stage_name}: {summarized_error}".strip())
                    logger.warning(
                        "Port stage reported failure",
                        extra=sanitize_log_extra(stage=stage_name, error=summarized_error, stats=result.get("stats")),
                    )
            except Exception as exc:
                sanitized_error = sanitize_for_log(str(exc), key="error")
                logger.exception(
                    "Port stage raised exception",
                    extra=sanitize_log_extra(stage=stage_name, error=sanitized_error),
                )
                run_stats["stages"][stage_name] = {
                    "success": False,
                    "error": sanitized_error,
                    "stats": {},
                }
                run_stats["errors"].append(f"{stage_name}: {sanitized_error}")

        run_stats["completed_at"] = datetime.utcnow().isoformat()
        run_stats["success"] = all(stage.get("success", False) for stage in run_stats["stages"].values())
        webhook_result = await self._dispatch_completion_webhook(run_stats)
        if webhook_result is not None:
            run_stats["webhook"] = webhook_result
        logger.info(
            "Port crawler run completed",
            extra=sanitize_log_extra(success=run_stats["success"], stage_count=len(run_stats["stages"]), errors=run_stats["errors"]),
        )
        return run_stats

    def _build_completion_webhook_payload(self, run_stats: dict[str, Any]) -> dict[str, Any]:
        """Build boundary-safe completion payload."""
        started_at = str(run_stats.get("started_at") or datetime.now(UTC).isoformat())
        completed_at = str(run_stats.get("completed_at") or datetime.now(UTC).isoformat())
        mode = str(run_stats.get("mode") or "daily")

        return {
            "job_id": f"port-{mode}-{started_at}",
            "scope": WEBHOOK_SCOPE,
            "completed_at": completed_at,
        }

    async def _dispatch_completion_webhook(self, run_stats: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch completion webhook with retry/backoff semantics."""
        webhook_url = str(getattr(settings, "CRAWLER_WEBHOOK_URL", "") or "").strip()
        webhook_secret = str(getattr(settings, "CRAWLER_WEBHOOK_SECRET", "") or "").strip()
        if not webhook_url or not webhook_secret:
            return None

        payload = self._build_completion_webhook_payload(run_stats)
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        signature = hmac.new(
            webhook_secret.encode("utf-8"),
            payload_json.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        signed_payload = dict(payload)
        signed_payload["signature"] = signature

        max_retries = max(int(getattr(settings, "CRAWLER_WEBHOOK_MAX_RETRIES", 3) or 3), 1)
        timeout_seconds = float(getattr(settings, "CRAWLER_WEBHOOK_TIMEOUT_SECONDS", 10.0) or 10.0)

        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    response = await client.post(webhook_url, json=signed_payload)
                if response.status_code < 400:
                    return {
                        "sent": True,
                        "attempts": attempt,
                        "status_code": response.status_code,
                    }

                last_error = f"HTTP {response.status_code}"
            except Exception as exc:
                last_error = str(exc)

            if attempt < max_retries:
                await asyncio.sleep(self._retry_delay_seconds(attempt))

        logger.warning(
            "Crawler completion webhook delivery failed",
            extra=sanitize_log_extra(error=last_error, retries=max_retries),
        )
        return {
            "sent": False,
            "attempts": max_retries,
            "error": sanitize_for_log(last_error or "unknown error"),
        }

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        """Exponential backoff with jitter: 100ms base, 2000ms cap, +/-25%."""
        base = min(0.1 * (2 ** max(attempt - 1, 0)), 2.0)
        jitter = random.uniform(0.75, 1.25)
        return base * jitter

    def _resolve_stage_runner(self, stage_name: str):
        mapping = {
            STAGE_EVENTS: self.run_events_stage,
            STAGE_METRICS: self.run_metrics_stage,
        }
        return mapping.get(stage_name)

    async def run_events_stage(
        self,
        *,
        project_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        db = self._session_factory()
        projects = self._load_projects(db, project_ids=project_ids)
        if not projects:
            db.close()
            return {"success": True, "skipped": True, "stats": {"reason": "No tracked projects found"}}

        stats = {
            "projects": len(projects),
            "updated_count": 0,
            "skipped_event_update": 0,
            "failed_projects": 0,
            "failure_reasons": [],
        }

        async def _run(stage: Any) -> None:
            for project in projects:
                try:
                    result = await stage.ingest_project(db, project)
                    stats["updated_count"] += int(result.updated_count)
                    if result.skipped_event_update:
                        stats["skipped_event_update"] += 1
                    if result.failure_reasons:
                        stats["failure_reasons"].append({"project": project.full_name, "reasons": result.failure_reasons})
                    # Write last_release back to the Project row if we got a date
                    if result.last_release_date is not None:
                        project.last_release = result.last_release_date
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    stats["failed_projects"] += 1
                    stats["failure_reasons"].append({"project": project.full_name, "reasons": [str(exc)]})
                    logger.warning(
                        "Events stage failed for project",
                        extra=sanitize_log_extra(stage=STAGE_EVENTS, project=project.full_name, error=str(exc)),
                    )

        try:
            if self._events_stage is not None:
                await _run(self._events_stage)
            else:
                async with self._github_client_factory() as client:
                    await _run(EventsStage(client))
            return {"success": True, "stats": stats}
        except Exception as exc:
            return {"success": False, "error": str(exc), "stats": stats}
        finally:
            db.close()

    async def run_metrics_stage(
        self,
        *,
        project_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        db = self._session_factory()
        projects = self._load_projects(db, project_ids=project_ids)
        if not projects:
            db.close()
            return {"success": True, "skipped": True, "stats": {"reason": "No tracked projects found"}}

        stage = self._metrics_stage or MetricsStage()
        snapshot_date = date.today()

        async def _fetch_payloads(client: Any) -> tuple[list[dict[str, Any]], int]:
            payloads: list[dict[str, Any]] = []
            failed = 0
            for project in projects:
                owner, repo = self._split_repo(project.full_name)
                response = await client.get_repo(owner, repo)
                if response.is_failed:
                    failed += 1
                    logger.warning(
                        "Metrics stage repo fetch failed",
                        extra=sanitize_log_extra(stage=STAGE_METRICS, repo=project.full_name, error=response.error),
                    )
                    continue

                data = response.data or {}
                payloads.append(
                    {
                        "project_id": project.id,
                        "stargazers_count": int(data.get("stargazers_count") or project.stars or 0),
                        "forks_count": int(data.get("forks_count") or project.forks or 0),
                        "open_issues_count": int(data.get("open_issues_count") or 0),
                        "contributors_count": int(data.get("subscribers_count") or project.contributors or 0),
                    }
                )
            return payloads, failed

        stats = {
            "projects": len(projects),
            "snapshot_date": snapshot_date.isoformat(),
            "processed": 0,
            "created": 0,
            "updated": 0,
            "failed": 0,
        }

        try:
            async with self._github_client_factory() as client:
                payloads, failed_fetches = await _fetch_payloads(client)
                stats["failed"] += failed_fetches
                result = stage.ingest_daily_metrics(db, metrics_payloads=payloads, snapshot_date=snapshot_date)
                stats["processed"] += int(result.get("processed", 0))
                stats["created"] += int(result.get("created", 0))
                stats["updated"] += int(result.get("updated", 0))
                stats["failed"] += int(result.get("failed", 0))
                db.commit()
            return {"success": True, "stats": stats}
        except Exception as exc:
            db.rollback()
            return {"success": False, "error": str(exc), "stats": stats}
        finally:
            db.close()

    @staticmethod
    def _load_projects(db: Any, *, project_ids: Sequence[int] | None) -> list[Project]:
        query = db.query(Project)
        if project_ids:
            query = query.filter(Project.id.in_(list(project_ids)))
        return list(query.all())

    @staticmethod
    def _split_repo(full_name: str) -> tuple[str, str]:
        owner, repo = full_name.split("/", 1)
        return owner.strip(), repo.strip()

    def _ensure_runtime_schema_compatibility(self) -> None:
        """Add required Port columns when running against older schemas."""

        statements = (
            "ALTER TABLE project_events ADD COLUMN IF NOT EXISTS event_types TEXT[]",
            "ALTER TABLE project_events ADD COLUMN IF NOT EXISTS bullets TEXT[]",
        )

        db = self._session_factory()
        try:
            from sqlalchemy import text
            for statement in statements:
                db.execute(text(statement))
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning(
                "Port runtime schema compatibility check failed",
                extra=sanitize_log_extra(stage="schema", error=str(exc)),
            )
        finally:
            db.close()
