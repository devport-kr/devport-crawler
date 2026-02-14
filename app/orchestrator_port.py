"""Port-domain orchestrator with dataset-isolated stage execution."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, date, datetime
import logging
from typing import Any, Awaitable, Callable, Sequence

from openai import AsyncOpenAI
from sqlalchemy import text

from app.config.database import SessionLocal
from app.config.settings import settings
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
from app.models.port import Port
from app.models.project import Project
from app.services.port.candidate_selector import CandidateSelector, RepoCandidate
from app.services.port.overview_sources import OverviewSourceAggregator
from app.services.port.overview_summarizer import PLACEHOLDER_TEXT, OverviewSummarizerService
from app.services.port.port_seed_catalog import PortSeed, get_default_port_seeds

logger = logging.getLogger(__name__)

STAGE_PROJECTS = "projects"
STAGE_EVENTS = "events"
STAGE_STAR_HISTORY = "star_history"
STAGE_METRICS = "metrics"
STAGE_OVERVIEWS = "overviews"

ALL_STAGES = (STAGE_PROJECTS, STAGE_EVENTS, STAGE_STAR_HISTORY, STAGE_METRICS, STAGE_OVERVIEWS)
DAILY_DEFAULT_STAGES = (STAGE_EVENTS, STAGE_METRICS, STAGE_OVERVIEWS)
DEFAULT_SEARCH_RESULTS_PER_PORT = 50

OVERVIEW_SYSTEM_MESSAGE = (
    "You write neutral, factual Korean technical summaries for software projects. "
    "Never use promotional language. Return valid JSON only."
)


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
        overview_llm_call: Callable[[str], Awaitable[Any]] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._github_client_factory = github_client_factory
        self._projects_stage = projects_stage
        self._events_stage = events_stage
        self._star_history_stage = star_history_stage
        self._metrics_stage = metrics_stage
        self._overview_stage = overview_stage
        self._overview_llm_call = overview_llm_call
        self._openai_client: AsyncOpenAI | None = None

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
        del checkpoints, mode, requested_metrics_days

        stage = self._projects_stage or ProjectsStage()
        db = self._session_factory()
        stats = {
            "ports": 0,
            "input": 0,
            "created": 0,
            "updated": 0,
            "failed": 0,
            "processed": 0,
            "ports_considered": 0,
            "ports_with_candidates": 0,
            "candidates_discovered": 0,
            "candidates_selected": 0,
        }
        try:
            payloads_by_port = repositories_by_port
            if payloads_by_port is None:
                ports = self._load_or_seed_ports(db)
                async with self._github_client_factory() as client:
                    payloads_by_port, discovery_stats = await self._discover_repositories_by_port(
                        client=client,
                        ports=ports,
                    )
                stats.update(discovery_stats)

            if project_ids:
                logger.info(
                    "Projects stage ignores project_ids filter during discovery",
                    extra=sanitize_log_extra(stage=STAGE_PROJECTS, project_ids=list(project_ids)),
                )

            if not payloads_by_port:
                return {
                    "success": True,
                    "skipped": True,
                    "stats": {**stats, "reason": "No repositories discovered for configured ports"},
                }

            for port_id, payloads in payloads_by_port.items():
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
                    summarizer = OverviewSummarizerService(llm_call=self._call_overview_llm)
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

    def _ensure_runtime_schema_compatibility(self) -> None:
        """Add required Port columns when running against older schemas."""

        statements = (
            "ALTER TABLE project_events ADD COLUMN IF NOT EXISTS event_types TEXT[]",
            "ALTER TABLE project_events ADD COLUMN IF NOT EXISTS bullets TEXT[]",
            "ALTER TABLE project_metrics_daily ADD COLUMN IF NOT EXISTS stars_week_delta INTEGER",
            "ALTER TABLE project_metrics_daily ADD COLUMN IF NOT EXISTS releases_30d INTEGER",
            "ALTER TABLE project_overviews ADD COLUMN IF NOT EXISTS highlights TEXT[]",
        )

        db = self._session_factory()
        try:
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

    @staticmethod
    def _load_or_seed_ports(db: Any) -> list[Port]:
        """Load tracked ports, seeding defaults when table is empty."""

        existing_ports = list(db.query(Port).all())
        if existing_ports:
            return existing_ports

        for seed in get_default_port_seeds():
            db.add(
                Port(
                    external_id=f"port:{seed.slug}",
                    port_number=seed.port_number,
                    slug=seed.slug,
                    name=seed.name,
                    description=seed.description,
                    accent_color=seed.accent_color,
                )
            )
        db.flush()
        return list(db.query(Port).all())

    async def _discover_repositories_by_port(
        self,
        *,
        client: GitHubPortClient,
        ports: Sequence[Port],
    ) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
        """Build per-port repository payloads via baseline + automatic search."""

        selector = CandidateSelector()
        seed_map = {seed.slug: seed for seed in get_default_port_seeds()}

        repositories_by_port: dict[int, list[dict[str, Any]]] = {}
        stats = {
            "ports_considered": 0,
            "ports_with_candidates": 0,
            "candidates_discovered": 0,
            "candidates_selected": 0,
        }

        for port in sorted(ports, key=lambda item: int(item.port_number or 0)):
            stats["ports_considered"] += 1
            seed_profile = self._resolve_seed_profile(port=port, seed=seed_map.get(str(port.slug)))

            baseline_payloads = await self._fetch_baseline_repositories(client=client, baseline_repos=seed_profile.baseline_repos)
            auto_payloads = await self._search_auto_candidates(client=client, keywords=seed_profile.keywords)

            payload_by_external_id: dict[str, dict[str, Any]] = {}
            for payload in [*baseline_payloads, *auto_payloads]:
                external_id = self._repo_external_id(payload)
                if external_id:
                    payload_by_external_id[external_id] = payload

            manual_candidates = [
                candidate
                for candidate in (self._repo_payload_to_candidate(payload) for payload in baseline_payloads)
                if candidate is not None
            ]
            auto_candidates = [
                candidate
                for candidate in (self._repo_payload_to_candidate(payload) for payload in auto_payloads)
                if candidate is not None and candidate.external_id not in {item.external_id for item in manual_candidates}
            ]

            stats["candidates_discovered"] += len(manual_candidates) + len(auto_candidates)
            selected = selector.select_candidates(
                manual_baseline=manual_candidates,
                auto_candidates=auto_candidates,
                relevance_keywords=seed_profile.keywords,
                target_count=getattr(settings, "PORT_PROJECT_TARGET_MAX", 20),
            )
            stats["candidates_selected"] += len(selected)

            selected_payloads: list[dict[str, Any]] = []
            for candidate in selected:
                payload = payload_by_external_id.get(candidate.external_id)
                if payload is not None:
                    selected_payloads.append(payload)

            if selected_payloads:
                repositories_by_port[int(port.id)] = selected_payloads
                stats["ports_with_candidates"] += 1

        return repositories_by_port, stats

    @staticmethod
    def _resolve_seed_profile(*, port: Port, seed: PortSeed | None) -> PortSeed:
        if seed is not None:
            return seed

        fallback_keywords = tuple(
            {
                token.strip().lower()
                for token in [str(port.slug or ""), str(port.name or ""), str(port.description or "")]
                for token in token.replace("/", " ").replace("-", " ").split()
                if token.strip()
            }
        )
        if not fallback_keywords:
            fallback_keywords = ("developer", "opensource")

        return PortSeed(
            slug=str(port.slug),
            name=str(port.name),
            port_number=int(port.port_number),
            description=str(port.description or ""),
            accent_color=str(port.accent_color or "#64748b"),
            keywords=fallback_keywords,
            baseline_repos=(),
        )

    async def _fetch_baseline_repositories(
        self,
        *,
        client: GitHubPortClient,
        baseline_repos: Sequence[str],
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for full_name in baseline_repos:
            if "/" not in full_name:
                continue
            owner, repo = full_name.split("/", 1)
            response = await client.get_repo(owner.strip(), repo.strip())
            if response.is_ok and isinstance(response.data, dict):
                payloads.append(response.data)
        return payloads

    async def _search_auto_candidates(
        self,
        *,
        client: GitHubPortClient,
        keywords: Sequence[str],
    ) -> list[dict[str, Any]]:
        terms = [token.strip() for token in keywords if token and token.strip()]
        if not terms:
            return []

        query = " ".join(terms[:5]) + " stars:>=100 archived:false"
        response = await client.search_repositories(
            query,
            page=1,
            per_page=DEFAULT_SEARCH_RESULTS_PER_PORT,
            sort="stars",
            order="desc",
        )
        if response.is_failed:
            logger.warning(
                "Repository search failed for port discovery",
                extra=sanitize_log_extra(stage=STAGE_PROJECTS, query=query, error=response.error),
            )
            return []

        if not response.data:
            return []
        return [payload for payload in response.data if isinstance(payload, dict)]

    @staticmethod
    def _repo_payload_to_candidate(payload: dict[str, Any]) -> RepoCandidate | None:
        full_name = str(payload.get("full_name") or "").strip()
        if "/" not in full_name:
            return None

        external_id = PortCrawlerOrchestrator._repo_external_id(payload)
        if external_id is None:
            return None

        pushed_at = PortCrawlerOrchestrator._parse_datetime(payload.get("pushed_at"))
        topics = payload.get("topics") if isinstance(payload.get("topics"), list) else []

        return RepoCandidate(
            external_id=external_id,
            full_name=full_name,
            description=str(payload.get("description") or ""),
            topics=tuple(str(topic).strip() for topic in topics if str(topic).strip()),
            stars=int(payload.get("stargazers_count") or 0),
            pushed_at=pushed_at,
            archived=bool(payload.get("archived") or False),
            disabled=bool(payload.get("disabled") or False),
        )

    @staticmethod
    def _repo_external_id(payload: dict[str, Any]) -> str | None:
        repo_id = payload.get("id")
        if isinstance(repo_id, int):
            return f"github:{repo_id}"

        full_name = str(payload.get("full_name") or "").strip().lower()
        if full_name:
            return f"github:{full_name}"
        return None

    @staticmethod
    def _parse_datetime(raw: Any) -> datetime | None:
        if not isinstance(raw, str) or not raw.strip():
            return None

        text = raw.strip().replace("Z", "+00:00")
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    async def _call_overview_llm(self, prompt: str) -> Any:
        if self._overview_llm_call is not None:
            return await self._overview_llm_call(prompt)
        return await self._default_overview_llm_call(prompt)

    async def _default_overview_llm_call(self, prompt: str) -> Any:
        if not settings.OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY not configured; using overview placeholder fallback",
                extra=sanitize_log_extra(stage=STAGE_OVERVIEWS),
            )
            return await self._fallback_overview_llm_call(prompt)

        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        try:
            response = await self._openai_client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": OVERVIEW_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=min(int(getattr(settings, "LLM_MAX_TOKENS", 6000)), 8000),
                reasoning_effort="low",
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise ValueError(f"overview llm call failed: {exc}") from exc
