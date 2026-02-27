"""Port-domain orchestrator with dataset-isolated stage execution."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, date, datetime
import hashlib
import hmac
import json
import logging
import random
from typing import Any, Awaitable, Callable, Sequence

import httpx

from app.config.database import SessionLocal
from app.config.settings import settings
from app.crawlers.port.client import GitHubPortClient, sanitize_for_log, sanitize_log_extra
from app.crawlers.port.events_stage import EventsStage
from app.crawlers.port.metrics_stage import MetricsStage
from app.crawlers.port.projects_stage import ProjectsStage
from app.models.port import Port
from app.models.project import Project
from app.services.port.candidate_selector import CandidateSelector, RepoCandidate

logger = logging.getLogger(__name__)

STAGE_PROJECTS = "projects"
STAGE_EVENTS = "events"
STAGE_METRICS = "metrics"

ALL_STAGES = (STAGE_PROJECTS, STAGE_EVENTS, STAGE_METRICS)
DAILY_DEFAULT_STAGES = (STAGE_EVENTS, STAGE_METRICS)
WEBHOOK_SCOPE = "GIT_REPO"

GLOBAL_BASELINE_REPOS = (
    "ollama/ollama",
    "vllm-project/vllm",
    "huggingface/text-generation-inference",
    "langchain-ai/langchain",
    "microsoft/autogen",
    "crewAIInc/crewAI",
    "openai/openai-python",
    "anthropics/anthropic-sdk-python",
    "vercel/ai",
    "mlflow/mlflow",
    "wandb/wandb",
    "bentoml/BentoML",
)

GLOBAL_KEYWORDS = (
    "llm", "inference", "model serving", "rag",
    "agent", "agentic", "tool-use", "multi-agent",
    "ai sdk", "ai cli", "code assistant", "copilot",
    "mlops", "experiment tracking", "model registry",
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
        metrics_stage: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._github_client_factory = github_client_factory
        self._projects_stage = projects_stage
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
            STAGE_PROJECTS: self.run_projects_stage,
            STAGE_EVENTS: self.run_events_stage,
            STAGE_METRICS: self.run_metrics_stage,
        }
        return mapping.get(stage_name)

    async def run_projects_stage(
        self,
        *,
        project_ids: Sequence[int] | None = None,
        checkpoints: dict[str, dict[str, Any]] | None = None,
        mode: str = "daily",
        requested_metrics_days: int = 3650,
        repositories_payloads: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del checkpoints, mode, requested_metrics_days

        stage = self._projects_stage or ProjectsStage()
        db = self._session_factory()
        stats = {
            "input": 0,
            "created": 0,
            "updated": 0,
            "failed": 0,
            "processed": 0,
            "candidates_discovered": 0,
            "candidates_selected": 0,
        }
        try:
            payloads = repositories_payloads
            if payloads is None:
                async with self._github_client_factory() as client:
                    baseline_payloads = await self._fetch_baseline_repositories(client=client, baseline_repos=GLOBAL_BASELINE_REPOS)
                    auto_payloads = await self._discover_repositories(client)

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
                selector = CandidateSelector()
                selected = selector.select_candidates(
                    manual_baseline=manual_candidates,
                    auto_candidates=auto_candidates,
                    relevance_keywords=GLOBAL_KEYWORDS,
                    target_count=getattr(settings, "PORT_PROJECT_GLOBAL_TARGET", 1000),
                )
                stats["candidates_selected"] += len(selected)

                payloads = []
                for candidate in selected:
                    payload = payload_by_external_id.get(candidate.external_id)
                    if payload is not None:
                        payloads.append(payload)

            if project_ids:
                logger.info(
                    "Projects stage ignores project_ids filter during discovery",
                    extra=sanitize_log_extra(stage=STAGE_PROJECTS, project_ids=list(project_ids)),
                )

            if not payloads:
                return {
                    "success": True,
                    "skipped": True,
                    "stats": {**stats, "reason": "No repositories discovered"},
                }

            result = stage.ingest_repositories(db, repositories=payloads)
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

    async def _discover_repositories(
        self,
        client: GitHubPortClient,
    ) -> list[dict[str, Any]]:
        """Automatically source new AI/trending repository payloads globally."""
        search_pages = 5
        results_per_page = 100
        keywords_per_query = 3
        chunks = [
            list(GLOBAL_KEYWORDS)[i : i + keywords_per_query]
            for i in range(0, len(GLOBAL_KEYWORDS), keywords_per_query)
        ]

        all_payloads: list[dict[str, Any]] = []

        for chunk in chunks:
            query = " OR ".join(f'"{k}"' for k in chunk) + " stars:>=500000 archived:false"
            for page in range(1, search_pages + 1):
                response = await client.search_repositories(
                    query,
                    page=page,
                    per_page=results_per_page,
                    sort="stars",
                    order="desc",
                )
                if response.is_ok and response.data:
                    all_payloads.extend(response.data)
                elif response.is_failed:
                    logger.warning(
                        "Repository search failed during global discovery",
                        extra=sanitize_log_extra(stage=STAGE_PROJECTS, query=query, error=response.error),
                    )
                    break

        return all_payloads

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
            per_page=100,
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
