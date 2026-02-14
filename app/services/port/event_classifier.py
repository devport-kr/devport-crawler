"""Event classification helpers for Port timeline ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.models.project_event import EventType


@dataclass(slots=True)
class ClassifiedEvent:
    """Normalized event payload ready for persistence."""

    event_types: list[str]
    impact_score: int
    is_security: bool
    is_breaking: bool


_KEYWORDS: dict[EventType, tuple[str, ...]] = {
    EventType.SECURITY: ("security", "cve", "vulnerability", "xss", "csrf", "auth bypass", "patch"),
    EventType.BREAKING: (
        "breaking",
        "breaking change",
        "migration required",
        "remove",
        "removed",
        "deprecated",
        "deprecate",
    ),
    EventType.FEATURE: ("feature", "add", "added", "new", "support"),
    EventType.FIX: ("fix", "fixed", "bug", "bugfix", "issue", "regression", "hotfix"),
    EventType.PERF: ("perf", "performance", "optimiz", "faster", "latency", "throughput"),
}


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def classify_event(*, title: str, body: str = "") -> ClassifiedEvent:
    """Map release/tag/changelog text to EventType-compatible values."""

    merged = f"{title}\n{body}".lower()
    event_types: list[str] = []

    for event_type in (
        EventType.SECURITY,
        EventType.BREAKING,
        EventType.FEATURE,
        EventType.FIX,
        EventType.PERF,
    ):
        if _contains_any(merged, _KEYWORDS[event_type]):
            event_types.append(event_type.value)

    if not event_types:
        event_types.append(EventType.MISC.value)

    is_security = EventType.SECURITY.value in event_types
    is_breaking = EventType.BREAKING.value in event_types

    impact_score = 1
    if is_security:
        impact_score += 4
    if is_breaking:
        impact_score += 3
    if EventType.FEATURE.value in event_types:
        impact_score += 2
    if EventType.FIX.value in event_types or EventType.PERF.value in event_types:
        impact_score += 1

    return ClassifiedEvent(
        event_types=event_types,
        impact_score=min(10, impact_score),
        is_security=is_security,
        is_breaking=is_breaking,
    )
