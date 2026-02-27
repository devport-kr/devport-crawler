"""Port-domain daily sync and backfill entrypoints."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from app.orchestrator_port import ALL_STAGES, DAILY_DEFAULT_STAGES, PortCrawlerOrchestrator


def normalize_stage_selector(stages: str | Sequence[str] | None, *, default: Sequence[str]) -> list[str]:
    """Normalize stage selector input into deterministic stage order."""
    if stages is None:
        return list(default)

    if isinstance(stages, str):
        requested = [part.strip() for part in stages.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in stages if str(part).strip()]

    if not requested:
        return list(default)

    allowed = set(ALL_STAGES)
    deduped: list[str] = []
    seen: set[str] = set()
    for stage in requested:
        if stage not in allowed or stage in seen:
            continue
        seen.add(stage)
        deduped.append(stage)
    return deduped or list(default)


async def run_port_daily_sync(
    *,
    orchestrator: PortCrawlerOrchestrator | None = None,
    stages: str | Sequence[str] | None = None,
    project_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run daily refresh for port events/metrics by default."""
    job_orchestrator = orchestrator or PortCrawlerOrchestrator()
    selected_stages = normalize_stage_selector(stages, default=DAILY_DEFAULT_STAGES)
    return await job_orchestrator.run_daily_sync(stages=selected_stages, project_ids=project_ids)


async def run_port_backfill(
    *,
    orchestrator: PortCrawlerOrchestrator | None = None,
    stages: str | Sequence[str] | None = None,
    project_ids: Sequence[int] | None = None,
    checkpoints: dict[str, dict[str, Any]] | None = None,
    requested_metrics_days: int = 3650,
) -> dict[str, Any]:
    """Run resumable full-history backfill with practical caps/checkpoints."""
    job_orchestrator = orchestrator or PortCrawlerOrchestrator()
    selected_stages = normalize_stage_selector(stages, default=ALL_STAGES)
    return await job_orchestrator.run_backfill(
        stages=selected_stages,
        project_ids=project_ids,
        checkpoints=checkpoints,
        requested_metrics_days=requested_metrics_days,
    )


def parse_project_ids(raw: Any) -> list[int] | None:
    """Parse optional project IDs from event payloads/query params."""
    if raw is None:
        return None

    if isinstance(raw, str):
        values: Iterable[Any] = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = [raw]

    parsed: list[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        try:
            parsed.append(int(text))
        except ValueError:
            continue
    return parsed or None
