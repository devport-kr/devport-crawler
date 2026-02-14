"""Source aggregation for project overview generation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from app.crawlers.port.contracts import FetchState


@dataclass(slots=True)
class SourceChunk:
    """A fetched and normalized source block."""

    label: str
    path: str
    url: str
    content: str


@dataclass(slots=True)
class OverviewSourcePayload:
    """Aggregated source payload used by overview stage."""

    source_url: str
    raw_text: str
    raw_hash: str
    links: list[dict[str, str]]
    fetched_at: datetime
    skipped: bool = False


class OverviewSourceAggregator:
    """Collect README/docs/model-library text into deterministic overview input."""

    README_PATHS = ("README.md", "readme.md", "README")
    DOC_PATHS = (
        "docs/index.md",
        "docs/README.md",
        "docs/getting-started.md",
        "docs/quickstart.md",
        "docs/installation.md",
        "docs/intro.md",
    )

    def __init__(self, github_client: Any) -> None:
        self._client = github_client

    async def collect(
        self,
        *,
        owner: str,
        repo: str,
        previous_raw_hash: str | None = None,
    ) -> OverviewSourcePayload:
        chunks: list[SourceChunk] = []

        readme_chunk = await self._fetch_first_path(owner=owner, repo=repo, paths=self.README_PATHS, label="README")
        if readme_chunk is not None:
            chunks.append(readme_chunk)

        linked_paths: set[str] = set()
        if readme_chunk is not None:
            linked_paths = self._extract_candidate_paths(owner=owner, repo=repo, markdown=readme_chunk.content)

        docs_paths = set(self.DOC_PATHS)
        docs_paths.update(linked_paths)

        for path in sorted(docs_paths):
            if readme_chunk is not None and path == readme_chunk.path:
                continue
            chunk = await self._fetch_path(owner=owner, repo=repo, path=path, label="DOC")
            if chunk is not None:
                chunks.append(chunk)

        normalized_chunks = self._dedupe_chunks(chunks)
        if not normalized_chunks:
            source_url = f"https://github.com/{owner}/{repo}"
            raw_text = ""
            raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            return OverviewSourcePayload(
                source_url=source_url,
                raw_text=raw_text,
                raw_hash=raw_hash,
                links=[],
                fetched_at=datetime.utcnow(),
                skipped=previous_raw_hash == raw_hash,
            )

        raw_text = self._compose_payload(normalized_chunks)
        raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        source_url = normalized_chunks[0].url
        links = [{"label": chunk.label, "url": chunk.url} for chunk in normalized_chunks]
        return OverviewSourcePayload(
            source_url=source_url,
            raw_text=raw_text,
            raw_hash=raw_hash,
            links=links,
            fetched_at=datetime.utcnow(),
            skipped=bool(previous_raw_hash and previous_raw_hash == raw_hash),
        )

    async def _fetch_first_path(
        self,
        *,
        owner: str,
        repo: str,
        paths: tuple[str, ...],
        label: str,
    ) -> SourceChunk | None:
        for path in paths:
            chunk = await self._fetch_path(owner=owner, repo=repo, path=path, label=label)
            if chunk is not None:
                return chunk
        return None

    async def _fetch_path(self, *, owner: str, repo: str, path: str, label: str) -> SourceChunk | None:
        result = await self._client.get_content(owner, repo, path)
        if result.state != FetchState.OK:
            return None
        content = (result.data or "").strip()
        if not content:
            return None

        extracted = self._extract_high_signal_sections(content)
        if not extracted:
            return None

        url = f"https://github.com/{owner}/{repo}/blob/main/{path}"
        return SourceChunk(label=label, path=path, url=url, content=extracted)

    def _extract_candidate_paths(self, *, owner: str, repo: str, markdown: str) -> set[str]:
        candidates: set[str] = set()
        for _, raw_url in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", markdown):
            parsed = self._parse_repo_relative_path(owner=owner, repo=repo, raw_url=raw_url)
            if parsed and self._is_high_signal_path(parsed):
                candidates.add(parsed)
        return candidates

    @staticmethod
    def _parse_repo_relative_path(*, owner: str, repo: str, raw_url: str) -> str | None:
        cleaned = raw_url.strip()
        if not cleaned or cleaned.startswith("#"):
            return None

        if cleaned.startswith(("http://", "https://")):
            parsed = urlparse(cleaned)
            path = parsed.path.strip("/")
            prefix = f"{owner}/{repo}/blob/"
            if not path.startswith(prefix):
                return None
            parts = path.split("/")
            if len(parts) < 5:
                return None
            return "/".join(parts[4:])

        if cleaned.startswith("/"):
            return cleaned.lstrip("/")

        return cleaned

    @staticmethod
    def _is_high_signal_path(path: str) -> bool:
        lowered = path.lower()
        if not lowered.endswith((".md", ".mdx", ".txt")):
            return False
        keywords = ("docs/", "model", "models", "library", "quickstart", "getting-started", "install", "guide")
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _extract_high_signal_sections(markdown: str) -> str:
        headings = re.split(r"(?m)^#{1,3}\s+", markdown)
        if len(headings) == 1:
            return markdown.strip()

        title = headings[0].strip()
        sections = headings[1:]
        selected: list[str] = []
        keywords = ("overview", "intro", "install", "quickstart", "usage", "example", "api", "architecture", "feature")

        for section in sections:
            lines = section.splitlines()
            if not lines:
                continue
            heading = lines[0].strip().lower()
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue
            if any(keyword in heading for keyword in keywords):
                selected.append(f"## {lines[0].strip()}\n{body}")

        if not selected:
            return markdown.strip()

        prefix = f"# {title}\n" if title else ""
        return (prefix + "\n\n".join(selected)).strip()

    @staticmethod
    def _dedupe_chunks(chunks: list[SourceChunk]) -> list[SourceChunk]:
        deduped: list[SourceChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", chunk.content).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(chunk)
        return deduped

    @staticmethod
    def _compose_payload(chunks: list[SourceChunk]) -> str:
        blocks: list[str] = []
        for chunk in chunks:
            blocks.append(f"### SOURCE: {chunk.label} ({chunk.path})\n{chunk.content.strip()}")
        return "\n\n".join(blocks).strip()
