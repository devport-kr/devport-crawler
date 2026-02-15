"""Neutral Korean deep-digest summarizer for project overviews."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError, field_validator

from app.config.settings import settings

PLACEHOLDER_TEXT = "요약 준비 중입니다. 원문 링크를 확인해주세요."


class OverviewSummaryPayload(BaseModel):
    """Validated summary schema for project overviews."""

    summary: str
    highlights: list[str]
    quickstart: str | None = None
    links: list[dict[str, str]]

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary is required")
        if len(cleaned) < 20:
            raise ValueError("summary too short")

        banned = ("최고", "완벽", "혁신", "놀라운", "강력한", "반드시", "!!!")
        if any(token in cleaned for token in banned):
            raise ValueError("summary must remain neutral and factual")
        return cleaned

    @field_validator("highlights")
    @classmethod
    def validate_highlights(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if len(cleaned) < 2:
            raise ValueError("highlights must include at least 2 items")
        return cleaned[:8]

    @field_validator("links")
    @classmethod
    def validate_links(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in value:
            url = (item.get("url") or "").strip()
            label = (item.get("label") or "").strip() or "Source"
            if not url:
                continue
            normalized.append({"label": label, "url": url})
        return normalized


class OverviewSummarizerService:
    """Generate policy-compliant overview summaries with retry and fallback."""

    def __init__(
        self,
        *,
        llm_call: Callable[[str], Awaitable[Any]],
        max_attempts: int | None = None,
        backoff_base_seconds: float | None = None,
        backoff_max_seconds: float | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._llm_call = llm_call
        self._max_attempts = max_attempts or settings.PORT_SUMMARY_MAX_ATTEMPTS
        self._backoff_base = backoff_base_seconds or settings.PORT_SUMMARY_BACKOFF_BASE_SECONDS
        self._backoff_max = backoff_max_seconds or settings.PORT_SUMMARY_BACKOFF_MAX_SECONDS
        self._sleep = sleeper

    async def summarize(
        self,
        *,
        project_name: str,
        source_markdown: str,
        links: list[dict[str, str]],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(project_name=project_name, source_markdown=source_markdown, links=links)

        for attempt in range(1, self._max_attempts + 1):
            try:
                raw_response = await self._llm_call(prompt)
                parsed = self._parse_response(raw_response)
                validated = OverviewSummaryPayload.model_validate(parsed)
                return validated.model_dump()
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
                if attempt == self._max_attempts:
                    break
                await self._sleep(self._next_backoff(attempt))

        return {
            "summary": PLACEHOLDER_TEXT,
            "highlights": [],
            "quickstart": None,
            "links": self._normalize_links(links),
        }

    def _build_prompt(self, *, project_name: str, source_markdown: str, links: list[dict[str, str]]) -> str:
        links_json = json.dumps(self._normalize_links(links), ensure_ascii=False)
        return (
            "역할: 한국 개발자를 위한 오픈소스 기술 문서 에디터.\n"
            "목표: 프로젝트의 핵심 가치, 적용 시나리오, 운영 관점 포인트를 빠르게 파악할 수 있는 요약을 작성하세요.\n"
            "아래 JSON 스키마를 반드시 지키세요: "
            '{"summary": "string", "highlights": ["string"], "quickstart": "string|null", '
            '"links": [{"label": "string", "url": "string"}]}.\n'
            "작성 규칙:\n"
            "- summary: 한국어 2~3문장, 핵심 기능/용도/도입 포인트 포함\n"
            "- highlights: 4~8개, 각 항목은 한 문장으로 구체적으로 작성\n"
            "- quickstart: 원문에 명시된 설치/실행 명령만 포함\n"
            "- 금지: LICENSE/CONTRIBUTING/배지 설명 위주 내용, 마케팅 문구, 과장 표현\n"
            "- 사실 기반으로만 작성하고 추측하지 마세요.\n\n"
            f"프로젝트: {project_name}\n"
            f"링크: {links_json}\n\n"
            "원문:\n"
            f"{source_markdown}"
        )

    @staticmethod
    def _parse_response(raw_response: Any) -> dict[str, Any]:
        if isinstance(raw_response, dict):
            return raw_response
        if not isinstance(raw_response, str):
            raise TypeError("LLM response must be dict or JSON string")

        content = raw_response.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)

    def _next_backoff(self, attempt: int) -> float:
        base = self._backoff_base * (2 ** (attempt - 1))
        bounded = min(base, self._backoff_max)
        jitter = min(0.1 * attempt, 1.0)
        return min(bounded + jitter, self._backoff_max)

    @staticmethod
    def _normalize_links(links: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in links:
            url = (item.get("url") or "").strip()
            label = (item.get("label") or "").strip() or "Source"
            if url:
                normalized.append({"label": label, "url": url})
        return normalized
