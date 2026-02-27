"""Compatibility tests for orchestrator webhook handoff contract."""

from app.orchestrator_port import PortCrawlerOrchestrator


def test_completion_payload_basic_contract() -> None:
    orchestrator = PortCrawlerOrchestrator()

    payload = orchestrator._build_completion_webhook_payload(
        {
            "mode": "daily",
            "started_at": "2026-02-16T00:00:00Z",
            "completed_at": "2026-02-16T00:00:30Z",
            "stages": {
                "events": {"success": True, "stats": {"updated_count": 5}},
                "metrics": {"success": True, "stats": {"processed": 3}},
            },
        }
    )

    assert payload["scope"] == "GIT_REPO"
    assert payload["job_id"].startswith("port-daily-")
    assert "completed_at" in payload
