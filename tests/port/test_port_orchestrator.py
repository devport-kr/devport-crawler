from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.jobs.port_sync import normalize_stage_selector, run_port_backfill, run_port_daily_sync
from app.orchestrator_port import ALL_STAGES, DAILY_DEFAULT_STAGES, PortCrawlerOrchestrator


class FakeOrchestrator(PortCrawlerOrchestrator):
    def __init__(self) -> None:
        super().__init__(session_factory=lambda: None)
        self.daily_calls = []
        self.backfill_calls = []

    async def run_daily_sync(self, *, stages=None, project_ids=None):
        self.daily_calls.append({"stages": stages, "project_ids": project_ids})
        return {"success": True, "mode": "daily", "stages_requested": list(stages or [])}

    async def run_backfill(self, *, stages=None, project_ids=None, checkpoints=None, requested_metrics_days=3650):
        self.backfill_calls.append(
            {
                "stages": stages,
                "project_ids": project_ids,
                "checkpoints": checkpoints,
                "requested_metrics_days": requested_metrics_days,
            }
        )
        return {"success": True, "mode": "backfill", "stages_requested": list(stages or [])}


def test_port_orchestrator_keeps_stage_isolation_on_failure(monkeypatch) -> None:
    orchestrator = PortCrawlerOrchestrator()
    call_order: list[str] = []

    async def fail_events(**_):
        call_order.append("events")
        raise RuntimeError("events unavailable")

    async def ok_metrics(**_):
        call_order.append("metrics")
        return {"success": True, "stats": {"processed": 2}}

    monkeypatch.setattr(orchestrator, "run_events_stage", fail_events)
    monkeypatch.setattr(orchestrator, "run_metrics_stage", ok_metrics)

    result = asyncio.run(orchestrator.run_daily_sync(stages=["events", "metrics"]))

    assert call_order == ["events", "metrics"]
    assert result["stages"]["events"]["success"] is False
    assert result["stages"]["metrics"]["success"] is True
    assert any("events:" in error for error in result["errors"])


def test_port_backfill_returns_checkpoint_and_cap_reason(monkeypatch) -> None:
    orchestrator = PortCrawlerOrchestrator()

    async def ok_metrics(**_):
        return {
            "success": True,
            "stats": {
                "cap_reasons": [{"scope": "metrics", "reason": "metrics history capped at 30 days"}],
            },
        }

    monkeypatch.setattr(orchestrator, "run_metrics_stage", ok_metrics)

    result = asyncio.run(orchestrator.run_backfill(stages=["metrics"], requested_metrics_days=30))
    stage_stats = result["stages"]["metrics"]["stats"]

    assert result["stages"]["metrics"]["success"] is True
    assert stage_stats["cap_reasons"][0]["reason"].startswith("metrics history capped")


def test_port_jobs_forward_stage_scope_and_backfill_options() -> None:
    orchestrator = FakeOrchestrator()

    asyncio.run(run_port_daily_sync(orchestrator=orchestrator, stages="events,metrics", project_ids=[100, 200]))
    asyncio.run(
        run_port_backfill(
            orchestrator=orchestrator,
            stages=["metrics"],
            project_ids=[300],
            checkpoints={"github:300": {"next_page": 2, "complete": False, "reached_cap": False}},
            requested_metrics_days=1200,
        )
    )

    assert orchestrator.daily_calls[0]["stages"] == ["events", "metrics"]
    assert orchestrator.daily_calls[0]["project_ids"] == [100, 200]
    assert orchestrator.backfill_calls[0]["stages"] == ["metrics"]
    assert orchestrator.backfill_calls[0]["requested_metrics_days"] == 1200


def test_stage_selector_defaults_and_filters_unknown_values() -> None:
    assert normalize_stage_selector(None, default=DAILY_DEFAULT_STAGES) == list(DAILY_DEFAULT_STAGES)
    assert normalize_stage_selector("events,metrics,invalid", default=DAILY_DEFAULT_STAGES) == ["events", "metrics"]
    assert normalize_stage_selector([], default=ALL_STAGES) == list(ALL_STAGES)


def test_projects_stage_runs_discovery_when_payload_not_injected(monkeypatch) -> None:
    class FakeDB:
        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeProjectsStage:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def ingest_repositories(self, _db: Any, *, repositories: list[dict[str, Any]]) -> dict[str, int]:
            self.calls.append({"repositories": repositories})
            return {
                "input": len(repositories),
                "created": len(repositories),
                "updated": 0,
                "failed": 0,
                "processed": len(repositories),
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

    fake_stage = FakeProjectsStage()
    orchestrator = PortCrawlerOrchestrator(
        session_factory=lambda: FakeDB(),
        github_client_factory=lambda: FakeClient(),
        projects_stage=fake_stage,
    )

    async def fake_discovery(client: Any) -> list[dict[str, Any]]:
        del client
        return [
            {
                "id": 101,
                "name": "demo",
                "full_name": "acme/demo",
                "html_url": "https://github.com/acme/demo",
                "description": "An awesome llm tool",
                "stargazers_count": 5000,
            }
        ]

    monkeypatch.setattr(orchestrator, "_discover_repositories", fake_discovery)
    monkeypatch.setattr("app.orchestrator_port.GLOBAL_BASELINE_REPOS", [])

    result = asyncio.run(orchestrator.run_projects_stage())

    assert result["success"] is True
    assert result["stats"]["created"] == 1
    assert len(fake_stage.calls) == 1
    assert fake_stage.calls[0]["repositories"][0]["id"] == 101


def test_completion_webhook_payload_contract() -> None:
    orchestrator = PortCrawlerOrchestrator()

    run_stats = {
        "mode": "daily",
        "started_at": "2026-02-16T00:00:00Z",
        "completed_at": "2026-02-16T00:01:00Z",
        "stages": {
            "events": {"success": True, "stats": {"updated_count": 3}},
            "metrics": {"success": True, "stats": {"processed": 5}},
        },
    }

    payload = orchestrator._build_completion_webhook_payload(run_stats)

    assert payload["scope"] == "GIT_REPO"
    assert payload["job_id"].startswith("port-daily-")
    assert "completed_at" in payload
