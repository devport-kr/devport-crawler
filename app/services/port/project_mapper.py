"""Contract-safe mapping helpers for Port project and metrics ingestion."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def build_project_external_id(repo_payload: dict[str, Any]) -> str:
    """Create stable external identity for `projects` upsert."""
    repo_id = repo_payload.get("id")
    if repo_id is not None:
        return f"github:{repo_id}"

    full_name = str(repo_payload.get("full_name") or "").strip().lower()
    if full_name:
        return f"github:{full_name}"
    raise ValueError("Repository payload missing both id and full_name")


def map_repo_to_project_row(
    *,
    repo_payload: dict[str, Any],
    existing: Any | None = None,
) -> dict[str, Any]:
    """Map GitHub payload into `projects` table contract while preserving valid existing fields."""

    now = datetime.utcnow()
    full_name = _pick_text(repo_payload.get("full_name"), getattr(existing, "full_name", None), required=True)
    fallback_name = full_name.split("/")[-1]

    return {
        "external_id": build_project_external_id(repo_payload),
        "name": _pick_text(repo_payload.get("name"), getattr(existing, "name", None), fallback=fallback_name, required=True),
        "full_name": full_name,
        "repo_url": _pick_text(
            repo_payload.get("html_url"),
            getattr(existing, "repo_url", None),
            fallback=f"https://github.com/{full_name}",
            required=True,
        ),
        "homepage_url": _pick_text(repo_payload.get("homepage"), getattr(existing, "homepage_url", None)),
        "description": _pick_text(repo_payload.get("description"), getattr(existing, "description", None)),
        "stars": _pick_int(repo_payload.get("stargazers_count"), getattr(existing, "stars", 0)),
        "forks": _pick_int(repo_payload.get("forks_count"), getattr(existing, "forks", 0)),
        "contributors": _pick_int(repo_payload.get("contributors_count"), getattr(existing, "contributors", 0)),
        "language": _pick_text(repo_payload.get("language"), getattr(existing, "language", None)),
        "language_color": _pick_text(repo_payload.get("language_color"), getattr(existing, "language_color", None)),
        "license": _pick_text(
            (repo_payload.get("license") or {}).get("spdx_id") if isinstance(repo_payload.get("license"), dict) else None,
            getattr(existing, "license", None),
        ),
        "tags": _pick_tags(repo_payload.get("topics"), getattr(existing, "tags", None)),
        "stars_week_delta": _pick_int(repo_payload.get("stars_week_delta"), getattr(existing, "stars_week_delta", 0)),
        "releases_30d": _pick_int(repo_payload.get("releases_30d"), getattr(existing, "releases_30d", 0)),
        "last_release": repo_payload.get("last_release") or getattr(existing, "last_release", None),
        "updated_at": now,
        "created_at": getattr(existing, "created_at", now),
    }


def map_metrics_to_daily_row(
    *,
    project_id: int,
    snapshot_date: date,
    metrics_payload: dict[str, Any],
    existing: Any | None = None,
) -> dict[str, Any]:
    """Map payload into `project_metrics_daily` contract."""

    return {
        "project_id": project_id,
        "date": snapshot_date,
        "stars": _pick_int(metrics_payload.get("stargazers_count"), getattr(existing, "stars", 0)),
        "forks": _pick_int(metrics_payload.get("forks_count"), getattr(existing, "forks", 0)),
        "open_issues": _pick_int(metrics_payload.get("open_issues_count"), getattr(existing, "open_issues", 0)),
        "contributors": _pick_int(metrics_payload.get("contributors_count"), getattr(existing, "contributors", 0)),
    }


def _pick_text(primary: Any, secondary: Any, *, fallback: str | None = None, required: bool = False) -> str | None:
    if isinstance(primary, str) and primary.strip():
        return primary.strip()
    if isinstance(secondary, str) and secondary.strip():
        return secondary.strip()
    if fallback is not None:
        return fallback
    if required:
        raise ValueError("Missing required textual field")
    return None


def _pick_int(primary: Any, secondary: Any) -> int:
    for value in (primary, secondary):
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return max(value, 0)
    return 0


def _pick_tags(primary: Any, secondary: Any) -> list[str] | None:
    for value in (primary, secondary):
        if isinstance(value, (list, tuple)):
            tags = [str(tag).strip() for tag in value if str(tag).strip()]
            if tags:
                return tags
    return None
