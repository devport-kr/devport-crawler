"""Event classification helpers for Port timeline ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import re
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
    EventType.SECURITY: (
        r"\bsecurity\b",
        r"\bcve-\d{4}-\d+\b",
        r"\bvulnerability\b",
        r"\bexploit\b",
        r"\bxss\b",
        r"\bcsrf\b",
        r"\bauth(?:entication)?\s+bypass\b",
    ),
    EventType.BREAKING: (
        r"\bbreaking\s+changes?\b",
        r"\bbreaking-change\b",
        r"\bbackward(?:s)?\s+incompatible\b",
        r"\bincompatible\s+change\b",
        r"\bmigration\s+required\b",
        r"\bdeprecated\b",
        r"\bdeprecation\b",
    ),
    EventType.FEATURE: (
        r"\bfeature(?:s)?\b",
        r"\bnew\b",
        r"\bintroduc(?:e|es|ed|ing)\b",
        r"\badd(?:s|ed|ing)?\b",
        r"\bsupport(?:s|ed|ing)?\b",
    ),
    EventType.FIX: (
        r"\bfix(?:es|ed|ing)?\b",
        r"\bbug(?:s|fix)?\b",
        r"\bhotfix\b",
        r"\bregression\b",
        r"\bresolve(?:s|d|ing)?\b",
        r"\bpatch(?:es|ed|ing)?\b",
    ),
    EventType.PERF: (
        r"\bperf(?:ormance)?\b",
        r"\boptimi[sz](?:e|es|ed|ing|ation)\b",
        r"\bfaster\b",
        r"\blatency\b",
        r"\bthroughput\b",
        r"\bmemory\s+usage\b",
    ),
}

_SECTION_HINTS: dict[EventType, tuple[str, ...]] = {
    EventType.SECURITY: (r"^#+\s*security",),
    EventType.BREAKING: (r"^#+\s*breaking",),
    EventType.FEATURE: (r"^#+\s*(feature|new)",),
    EventType.FIX: (r"^#+\s*(fix|bug)",),
    EventType.PERF: (r"^#+\s*(perf|performance)",),
}

_PRIORITY: dict[EventType, int] = {
    EventType.SECURITY: 0,
    EventType.BREAKING: 1,
    EventType.FEATURE: 2,
    EventType.FIX: 3,
    EventType.PERF: 4,
}


def _count_matches(text: str, patterns: Iterable[str]) -> int:
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
    return total


def classify_event(*, title: str, body: str = "") -> ClassifiedEvent:
    """Map release/tag/changelog text to EventType-compatible values."""

    title_text = title or ""
    body_text = body or ""

    scores: dict[EventType, int] = {}
    for event_type in (EventType.SECURITY, EventType.BREAKING, EventType.FEATURE, EventType.FIX, EventType.PERF):
        keyword_score = _count_matches(title_text, _KEYWORDS[event_type]) * 2
        keyword_score += _count_matches(body_text, _KEYWORDS[event_type])
        section_score = _count_matches(body_text, _SECTION_HINTS[event_type]) * 2
        scores[event_type] = keyword_score + section_score

    ranked = sorted(scores.items(), key=lambda item: (item[1], -_PRIORITY[item[0]]), reverse=True)
    selected: list[EventType] = [item[0] for item in ranked if item[1] >= 2][:2]

    if scores[EventType.SECURITY] > 0 and EventType.SECURITY not in selected:
        selected.insert(0, EventType.SECURITY)
        selected = selected[:2]

    if not selected:
        top_type, top_score = ranked[0]
        if top_score > 0:
            selected = [top_type]
        else:
            selected = [EventType.MISC]

    event_types = [event_type.value for event_type in selected]
    is_security = EventType.SECURITY in selected
    is_breaking = EventType.BREAKING in selected

    impact_score = 1
    if is_security:
        impact_score += 4
    if is_breaking:
        impact_score += 3
    if EventType.FEATURE in selected:
        impact_score += 2
    if EventType.FIX in selected or EventType.PERF in selected:
        impact_score += 1

    return ClassifiedEvent(
        event_types=event_types,
        impact_score=min(10, impact_score),
        is_security=is_security,
        is_breaking=is_breaking,
    )
