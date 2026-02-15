"""Wiki generation stage that produces Core-6 snapshots."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.crawlers.port.client import GitHubPortClient, sanitize_log_extra
from app.crawlers.wiki.contracts import WikiSection, WikiSnapshot, ReadinessMetadata, calculate_hidden_sections
from app.models.project import Project
from app.models.project_wiki_snapshot import ProjectWikiSnapshot
from app.services.wiki.wiki_prompt_builder import WikiPromptBuilder
from app.services.wiki.readiness_evaluator import ReadinessEvaluator

logger = logging.getLogger(__name__)


class WikiStage:
    """Orchestrates wiki generation with Core-6 sections, adaptive depth, and readiness gates."""

    def __init__(
        self,
        *,
        github_client: GitHubPortClient | None = None,
        prompt_builder: WikiPromptBuilder | None = None,
        readiness_evaluator: ReadinessEvaluator | None = None,
        min_stars_threshold: int = 100,
    ) -> None:
        self._github_client = github_client
        self._prompt_builder = prompt_builder or WikiPromptBuilder()
        self._readiness_evaluator = readiness_evaluator or ReadinessEvaluator(min_stars=min_stars_threshold)
        self._min_stars_threshold = min_stars_threshold

    async def run(
        self,
        db: Session,
        projects: list[Project],
    ) -> dict[str, Any]:
        """Generate and persist wiki snapshots for eligible projects.
        
        Args:
            db: Database session for persistence.
            projects: List of projects to generate wikis for.
            
        Returns:
            Stats dict with success/failure counts.
        """
        stats = {
            "projects_total": len(projects),
            "data_ready": 0,
            "excluded": 0,
            "created": 0,
            "refreshed": 0,
            "failed": 0,
        }

        for project in projects:
            try:
                # Check readiness first
                readiness_meta = await self._evaluate_readiness(project)
                if not readiness_meta.passes_top_star_gate:
                    stats["excluded"] += 1
                    logger.info(
                        "Wiki excluded - below star threshold",
                        extra=sanitize_log_extra(
                            project=project.full_name,
                            stars=readiness_meta.actual_stars,
                            threshold=readiness_meta.min_stars_threshold,
                        ),
                    )
                    continue

                # Generate wiki sections
                snapshot = await self._generate_snapshot(project, readiness_meta)

                # Persist with upsert
                is_new = self._upsert_snapshot(db, snapshot)
                if is_new:
                    stats["created"] += 1
                else:
                    stats["refreshed"] += 1

                if snapshot.is_data_ready:
                    stats["data_ready"] += 1

                db.commit()

            except Exception as exc:
                db.rollback()
                stats["failed"] += 1
                logger.warning(
                    f"Wiki generation failed for {project.full_name}: {type(exc).__name__}: {str(exc)}",
                    extra=sanitize_log_extra(project=project.full_name, error=str(exc)),
                )

        return {"success": True, "stats": stats}

    async def _evaluate_readiness(self, project: Project) -> ReadinessMetadata:
        """Evaluate whether project meets data-ready criteria."""
        return await self._readiness_evaluator.evaluate(project)

    async def _generate_snapshot(
        self,
        project: Project,
        readiness_meta: ReadinessMetadata,
    ) -> WikiSnapshot:
        """Generate complete Core-6 wiki snapshot for a project.
        
        Args:
            project: Project model with metadata.
            readiness_meta: Pre-computed readiness gates.
            
        Returns:
            Complete wiki snapshot with all sections.
        """
        # Build sections with progressive disclosure
        what_section = await self._build_what_section(project)
        how_section = await self._build_how_section(project)
        architecture_section = await self._build_architecture_section(project)
        activity_section = await self._build_activity_section(project)
        releases_section = await self._build_releases_section(project)
        chat_section = await self._build_chat_section(project)

        # Create snapshot
        snapshot = WikiSnapshot(
            project_external_id=project.external_id,
            generated_at=datetime.now(UTC).isoformat(),
            what=what_section,
            how=how_section,
            architecture=architecture_section,
            activity=activity_section,
            releases=releases_section,
            chat=chat_section,
            is_data_ready=readiness_meta.passes_top_star_gate and readiness_meta.has_sufficient_readme,
            hidden_sections=(),  # Will be calculated below
            readiness_metadata=readiness_meta,
        )

        # Calculate hidden sections based on readiness
        hidden = calculate_hidden_sections(snapshot)
        from dataclasses import replace
        snapshot = replace(snapshot, hidden_sections=hidden)

        return snapshot

    async def _build_what_section(self, project: Project) -> WikiSection:
        """Build 'What this project is' section with purpose and target users."""
        summary = f"{project.name} - {project.description or 'No description available'}"
        
        deep_dive = await self._prompt_builder.build_what_explanation(
            name=project.name,
            description=project.description or "",
            topics=getattr(project, "topics", []),
        )

        return WikiSection(
            summary=summary[:200],  # Keep summary concise
            deep_dive_markdown=deep_dive,
            default_expanded=True,
        )

    async def _build_how_section(self, project: Project) -> WikiSection:
        """Build 'How it works' section with key concepts and workflow."""
        deep_dive = await self._prompt_builder.build_how_explanation(
            name=project.name,
            description=project.description or "",
        )

        return WikiSection(
            summary="Key concepts, workflow, and usage patterns for this project.",
            deep_dive_markdown=deep_dive,
            default_expanded=False,
        )

    async def _build_architecture_section(self, project: Project) -> WikiSection:
        """Build architecture/codebase section with adaptive depth."""
        deep_dive = await self._prompt_builder.build_architecture_explanation(
            name=project.name,
            language=getattr(project, "language", "Unknown"),
        )

        return WikiSection(
            summary="Technical architecture, codebase structure, and design patterns.",
            deep_dive_markdown=deep_dive,
            default_expanded=False,
            generated_diagram_dsl=None,  # Diagram generation deferred
        )

    async def _build_activity_section(self, project: Project) -> WikiSection:
        """Build activity section emphasizing last 12 months."""
        return WikiSection(
            summary=f"Repository activity and contributor insights for {project.name}.",
            deep_dive_markdown="Activity timeline with last 12 months emphasis (data from activity_release_stage).",
            default_expanded=False,
        )

    async def _build_releases_section(self, project: Project) -> WikiSection:
        """Build releases section with timeline + all tags."""
        return WikiSection(
            summary=f"Release timeline and version tags for {project.name}.",
            deep_dive_markdown="Complete release/tag timeline (data from activity_release_stage).",
            default_expanded=False,
        )

    async def _build_chat_section(self, project: Project) -> WikiSection:
        """Build chat context section for Q&A grounding."""
        return WikiSection(
            summary="Interactive chat with repository-specific context.",
            deep_dive_markdown=f"Chat context payload for {project.name} Q&A.",
            default_expanded=False,
        )

    def _upsert_snapshot(self, db: Session, snapshot: WikiSnapshot) -> bool:
        """Persist snapshot with refresh-safe upsert.
        
        Args:
            db: Database session.
            snapshot: Wiki snapshot to persist.
            
        Returns:
            True if new record created, False if existing record updated.
        """
        existing = (
            db.query(ProjectWikiSnapshot)
            .filter(ProjectWikiSnapshot.project_external_id == snapshot.project_external_id)
            .first()
        )

        section_to_dict = lambda s: {
            "summary": s.summary,
            "deep_dive_markdown": s.deep_dive_markdown,
            "default_expanded": s.default_expanded,
            "generated_diagram_dsl": s.generated_diagram_dsl,
        }

        now = datetime.now(UTC)
        payload = {
            "project_external_id": snapshot.project_external_id,
            "generated_at": datetime.fromisoformat(snapshot.generated_at),
            "what_section": section_to_dict(snapshot.what),
            "how_section": section_to_dict(snapshot.how),
            "architecture_section": section_to_dict(snapshot.architecture),
            "activity_section": section_to_dict(snapshot.activity),
            "releases_section": section_to_dict(snapshot.releases),
            "chat_section": section_to_dict(snapshot.chat),
            "is_data_ready": snapshot.is_data_ready,
            "hidden_sections": list(snapshot.hidden_sections),
            "readiness_metadata": asdict(snapshot.readiness_metadata),
        }

        if existing:
            # Update existing
            for key, value in payload.items():
                setattr(existing, key, value)
            setattr(existing, "updated_at", now)
            db.flush()
            return False
        else:
            # Create new
            payload["created_at"] = now
            payload["updated_at"] = now
            db.add(ProjectWikiSnapshot(**payload))
            db.flush()
            return True
