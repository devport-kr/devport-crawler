"""Port-domain orchestrator with dataset-isolated stage execution."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import logging
from typing import Any, Callable, Sequence

from app.config.database import SessionLocal
from app.crawlers.port.client import GitHubPortClient, sanitize_for_log, sanitize_log_extra
from app.crawlers.port.events_stage import EventsStage
from app.crawlers.port.metrics_stage import MetricsStage
from app.crawlers.port.overview_stage import (
    OverviewProjectRef,
    ProjectOverviewStage,
    SQLAlchemyOverviewRepository,
)
from app.crawlers.port.projects_stage import ProjectsStage
from app.crawlers.port.star_history_stage import StarHistoryCheckpoint, StarHistoryStage
from app.models.project import Project
from app.services.port.overview_sources import OverviewSourceAggregator
from app.services.port.overview_summarizer import PLACEHOLDER_TEXT, OverviewSummarizerService

logger = logging.getLogger(__name__)

STAGE_PROJECTS = "projects"
STAGE_EVENTS = "events"
STAGE_STAR_HISTORY = "star_history"
STAGE_METRICS = "metrics"
STAGE_OVERVIEWS = "overviews"

ALL_STAGES = (STAGE_PROJECTS, STAGE_EVENTS, STAGE_STAR_HISTORY, STAGE_METRICS, STAGE_OVERVIEWS)
DAILY_DEFAULT_STAGES = (STAGE_EVENTS, STAGE_METRICS, STAGE_OVERVIEWS)


class PortCrawlerOrchestrator:
    """Coordinates isolated execution of port-domain ingestion stages."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = SessionLocal,
        github_client_factory: Callable[[], Any] = GitHubPortClient,
        projects_stage: Any | None = None,
        events_stage: Any | None = None,
        star_history_stage: Any | None = None,
        metrics_stage: Any | None = None,
        overview_stage: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._github_client_factory = github_client_factory
        self._projects_stage = projects_stage
        self._events_stage = events_stage
        self._star_history_stage = star_history_stage
        self._metrics_stage = metrics_stage
        self._overview_stage = overview_stage

    async def run_daily_sync(
        self,
        *,
        stages: Sequence[str] | None = None,
        project_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        selected_stages = tuple(stages or DAILY_DEFAULT_STAGES)
        return await self._run(mode="daily", stages=selected_stages, project_ids=project_ids)

    async def run_backfill(
        self,
        *,
        stages: Sequence[str] | None = None,
        project_ids: Sequence[int] | None = None,
        checkpoints: dict[str, dict[str, Any]] | None = None,
        requested_metrics_days: int = 3650,
    ) -> dict[str, Any]:
        selected_stages = tuple(stages or ALL_STAGES)
        return await self._run(
            mode="backfill",
            stages=selected_stages,
            project_ids=project_ids,
            checkpoints=checkpoints,
            requested_metrics_days=requested_metrics_days,
        )

    async def _run(
        self,
        *,
        mode: str,
        stages: Sequence[str],
        project_ids: Sequence[int] | None = None,
        checkpoints: dict[str, dict[str, Any]] | None = None,
        requested_metrics_days: int = 3650,
    ) -> dict[str, Any]:
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
                    checkpoints=checkpoints or {},
                    mode=mode,
                    requested_metrics_days=requested_metrics_days,
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
        logger.info(
            "Port crawler run completed",
            extra=sanitize_log_extra(success=run_stats["success"], stage_count=len(run_stats["stages"]), errors=run_stats["errors"]),
        )
        return run_stats

    def _resolve_stage_runner(self, stage_name: str):
        mapping = {
            STAGE_PROJECTS: self.run_projects_stage,
            STAGE_EVENTS: self.run_events_stage,
            STAGE_STAR_HISTORY: self.run_star_history_stage,
            STAGE_METRICS: self.run_metrics_stage,
            STAGE_OVERVIEWS: self.run_overview_stage,
        }
        return mapping.get(stage_name)

    async def run_projects_stage(
        self,
        *,
        project_ids: Sequence[int] | None = None,
        checkpoints: dict[str, dict[str, Any]] | None = None,
        mode: str = "daily",
        requested_metrics_days: int = 3650,
        repositories_by_port: dict[int, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        del project_ids, checkpoints, mode, requested_metrics_days
        if not repositories_by_port:
            return {
                "success": True,
                "skipped": True,
                "stats": {"reason": "No repositories_by_port payload provided"},
            }

        stage = self._projects_stage or ProjectsStage()
        db = self._session_factory()
        stats = {"ports": 0, "input": 0, "created": 0, "updated": 0, "failed": 0, "processed": 0}
        try:
            for port_id, payloads in repositories_by_port.items():
                result = stage.ingest_repositories(db, port_id=int(port_id), repositories=payloads)
                stats["ports"] += 1
                stats["input"] += int(result.get("input", 0))
                stats["created"] += int(result.get("created", 0))
                stats["updated"] += int(result.get("updated", 0))
                stats["failed"] += int(result.get("failed", 0))
                stats["processed"] += int(result.get("processed", 0))
            db.commit()
            return {"success": True, "stats": stats}
        except Exception as exc:
            db.rollback()
            return {"success": False, "error": str(exc), "stats": stats}
        finally:
            db.close()

    async def run_events_stage(
        self,
        *,
        project_ids: Sequence[int] | None = None,
        checkpoints: dict[str, dict[str, Any]] | None = None,
        mode: str = "daily",
        requested_metrics_days: int = 3650,
    ) -> dict[str, Any]:
        del checkpoints, mode, requested_metrics_days
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

    async def run_star_history_stage(
        self,
        *,
        project_ids: Sequence[int] | None = None,
        checkpoints: dict[str, dict[str, Any]] | None = None,
        mode: str = "daily",
        requested_metrics_days: int = 3650,
    ) -> dict[str, Any]:
        del mode, requested_metrics_days
        db = self._session_factory()
        projects = self._load_projects(db, project_ids=project_ids)
        if not projects:
            db.close()
            return {"success": True, "skipped": True, "stats": {"reason": "No tracked projects found"}}

        checkpoint_payloads = checkpoints or {}
        stats = {
            "projects": len(projects),
            "stored_points": 0,
            "source_points": 0,
            "fetched_pages": 0,
            "failed_projects": 0,
            "checkpoints": {},
            "cap_reasons": [],
        }

        async def _run(stage: Any) -> None:
            for project in projects:
                incoming = checkpoint_payloads.get(project.external_id)
                checkpoint = StarHistoryCheckpoint(**incoming) if incoming else None
                try:
                    result = await stage.ingest_project(db, project, checkpoint=checkpoint)
                    stats["stored_points"] += int(result.stored_points)
                    stats["source_points"] += int(result.source_points)
                    stats["fetched_pages"] += int(result.fetched_pages)
                    stats["checkpoints"][project.external_id] = asdict(result.checkpoint)
                    if result.checkpoint.reached_cap:
                        stats["cap_reasons"].append(
                            {
                                "project": project.full_name,
                                "reason": f"stargazer pages capped at {stage._max_pages}",
                            }
                        )
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    stats["failed_projects"] += 1
                    stats["cap_reasons"].append({"project": project.full_name, "reason": f"failed: {exc}"})
                    logger.warning(
                        "Star history stage failed for project",
                        extra=sanitize_log_extra(stage=STAGE_STAR_HISTORY, project=project.full_name, error=str(exc)),
                    )

        try:
            if self._star_history_stage is not None:
                await _run(self._star_history_stage)
            else:
                async with self._github_client_factory() as client:
                    await _run(StarHistoryStage(client))
            return {"success": True, "stats": stats}
        except Exception as exc:
            return {"success": False, "error": str(exc), "stats": stats}
        finally:
            db.close()

    async def run_metrics_stage(
        self,
        *,
        project_ids: Sequence[int] | None = None,
        checkpoints: dict[str, dict[str, Any]] | None = None,
        mode: str = "daily",
        requested_metrics_days: int = 3650,
    ) -> dict[str, Any]:
        del checkpoints
        db = self._session_factory()
        projects = self._load_projects(db, project_ids=project_ids)
        if not projects:
            db.close()
            return {"success": True, "skipped": True, "stats": {"reason": "No tracked projects found"}}

        stage = self._metrics_stage or MetricsStage()
        snapshot_date = date.today()
        dates = [snapshot_date]
        cap_reasons: list[dict[str, str]] = []
        if mode == "backfill":
            capped_days = stage.capped_backfill_days(requested_metrics_days)
            dates = stage.backfill_dates(end_date=snapshot_date, requested_days=requested_metrics_days)
            if capped_days < requested_metrics_days:
                cap_reasons.append(
                    {
                        "scope": "metrics",
                        "reason": f"metrics history capped at {capped_days} days",
                    }
                )

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
                        "stars_week_delta": int(project.stars_week_delta or 0),
                        "releases_30d": int(project.releases_30d or 0),
                    }
                )
            return payloads, failed

        stats = {
            "projects": len(projects),
            "dates": [d.isoformat() for d in dates],
            "processed": 0,
            "created": 0,
            "updated": 0,
            "failed": 0,
            "cap_reasons": cap_reasons,
        }

        try:
            async with self._github_client_factory() as client:
                payloads, failed_fetches = await _fetch_payloads(client)
                stats["failed"] += failed_fetches
                for metric_date in dates:
                    result = stage.ingest_daily_metrics(db, metrics_payloads=payloads, snapshot_date=metric_date)
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

    async def run_overview_stage(
        self,
        *,
        project_ids: Sequence[int] | None = None,
        checkpoints: dict[str, dict[str, Any]] | None = None,
        mode: str = "daily",
        requested_metrics_days: int = 3650,
    ) -> dict[str, Any]:
        del checkpoints, mode, requested_metrics_days
        db = self._session_factory()
        projects = self._load_projects(db, project_ids=project_ids)
        if not projects:
            db.close()
            return {"success": True, "skipped": True, "stats": {"reason": "No tracked projects found"}}

        refs = self._build_overview_refs(projects)
        if not refs:
            db.close()
            return {"success": True, "skipped": True, "stats": {"reason": "No valid project names"}}

        try:
            if self._overview_stage is not None:
                stage = self._overview_stage
                stage_stats = await stage.run(refs)
            else:
                async with self._github_client_factory() as client:
                    source_aggregator = OverviewSourceAggregator(client)
                    summarizer = OverviewSummarizerService(llm_call=self._fallback_overview_llm_call)
                    repository = SQLAlchemyOverviewRepository(db)
                    stage = ProjectOverviewStage(
                        source_aggregator=source_aggregator,
                        summarizer=summarizer,
                        repository=repository,
                    )
                    stage_stats = await stage.run(refs)

            db.commit()
            return {"success": True, "stats": asdict(stage_stats)}
        except Exception as exc:
            db.rollback()
            return {"success": False, "error": str(exc), "stats": {}}
        finally:
            db.close()

    @staticmethod
    async def _fallback_overview_llm_call(_: str) -> dict[str, Any]:
        return {
            "summary": PLACEHOLDER_TEXT,
            "highlights": [],
            "quickstart": None,
            "links": [],
        }

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

    @staticmethod
    def _build_overview_refs(projects: Sequence[Project]) -> list[OverviewProjectRef]:
        refs: list[OverviewProjectRef] = []
        for project in projects:
            if "/" not in project.full_name:
                continue
            owner, repo = project.full_name.split("/", 1)
            refs.append(
                OverviewProjectRef(
                    project_id=int(project.id),
                    owner=owner.strip(),
                    repo=repo.strip(),
                    project_name=project.name,
                )
            )
        return refs
