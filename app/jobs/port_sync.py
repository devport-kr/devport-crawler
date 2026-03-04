"""Port-domain daily sync entrypoint."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from app.orchestrator_port import DAILY_DEFAULT_STAGES, PortCrawlerOrchestrator


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

    allowed = set(DAILY_DEFAULT_STAGES)
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
    """Run daily refresh for port events/metrics."""
    job_orchestrator = orchestrator or PortCrawlerOrchestrator()
    selected_stages = normalize_stage_selector(stages, default=DAILY_DEFAULT_STAGES)
    return await job_orchestrator.run_daily_sync(stages=selected_stages, project_ids=project_ids)


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
