"""Resilient async GitHub client for Port-domain ingestion."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from typing import Any, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config.settings import settings
from app.crawlers.port.contracts import (
    ContentContract,
    FetchResult,
    FetchState,
    ReleaseContract,
    RepoContract,
    StargazerContract,
    TagContract,
)

logger = logging.getLogger(__name__)

_REDACTED_VALUE = "***REDACTED***"
_SENSITIVE_KEYS = (
    "authorization",
    "token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "cookie",
    "session",
)
_PAYLOAD_KEYS = ("body", "raw", "content", "payload", "response", "notes", "markdown", "blob")
_TOKEN_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(token\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)(access_token=)[^&\s]+"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)(secret\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)(cookie\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)(session\s*[=:]\s*)[^\s,;]+"),
)


def sanitize_for_log(value: Any, *, key: Optional[str] = None) -> Any:
    """Return a recursively sanitized copy of log payloads."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            field = str(raw_key)
            if _contains_keyword(field, _SENSITIVE_KEYS):
                sanitized[field] = _REDACTED_VALUE
                continue
            if _contains_keyword(field, _PAYLOAD_KEYS) and isinstance(raw_value, str):
                sanitized[field] = _redact_payload(raw_value)
                continue
            sanitized[field] = sanitize_for_log(raw_value, key=field)
        return sanitized

    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_log(item, key=key) for item in value]

    if isinstance(value, str):
        redacted = _redact_text(value)
        if key and _contains_keyword(key, _PAYLOAD_KEYS):
            return _redact_payload(redacted)
        return redacted

    return value


def sanitize_log_extra(**kwargs: Any) -> dict[str, Any]:
    """Helper for `extra=` payloads in structured logging."""

    return {key: sanitize_for_log(value, key=key) for key, value in kwargs.items()}


def _contains_keyword(field_name: str, keywords: tuple[str, ...]) -> bool:
    lowered = field_name.lower()
    return any(keyword in lowered for keyword in keywords)


def _redact_payload(raw: str) -> str:
    trimmed = raw.strip()
    if not trimmed:
        return ""
    return f"<redacted payload ({len(raw)} chars)>"


def _redact_text(raw: str) -> str:
    redacted = raw
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub(rf"\1{_REDACTED_VALUE}", redacted)
    return redacted


class _RateLimitRetryableError(Exception):
    """Retryable rate-limit signal for tenacity."""


class GitHubPortClient:
    """Typed GitHub API client with ETag and rate-limit resilience."""

    BASE_URL = "https://api.github.com"
    API_VERSION = "2022-11-28"
    ACCEPT_JSON = "application/vnd.github+json"
    ACCEPT_STARGAZERS = "application/vnd.github.star+json"

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_base_seconds: Optional[float] = None,
        backoff_max_seconds: Optional[float] = None,
        rate_limit_buffer_seconds: Optional[int] = None,
        base_url: Optional[str] = None,
        transport: Optional[Any] = None,
    ) -> None:
        self._token = token or settings.GITHUB_TOKEN
        self._timeout_seconds = timeout_seconds or getattr(settings, "PORT_GITHUB_TIMEOUT_SECONDS", 30.0)
        self._max_retries = max_retries or getattr(settings, "PORT_GITHUB_MAX_RETRIES", 3)
        self._backoff_base_seconds = backoff_base_seconds or getattr(settings, "PORT_GITHUB_BACKOFF_BASE_SECONDS", 1.0)
        self._backoff_max_seconds = backoff_max_seconds or getattr(settings, "PORT_GITHUB_BACKOFF_MAX_SECONDS", 16.0)
        self._rate_limit_buffer_seconds = rate_limit_buffer_seconds or getattr(settings, "PORT_GITHUB_RATE_LIMIT_BUFFER_SECONDS", 2)
        self._base_url = base_url or self.BASE_URL
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GitHubPortClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_repo(self, owner: str, repo: str, etag: Optional[str] = None) -> RepoContract:
        path = f"/repos/{owner}/{repo}"
        return await self._fetch_json_contract(path, etag=etag)

    async def search_repositories(
        self,
        query: str,
        *,
        page: int = 1,
        per_page: int = 50,
        sort: str = "stars",
        order: str = "desc",
        etag: Optional[str] = None,
    ) -> FetchResult[list[dict[str, Any]]]:
        """Search GitHub repositories using query syntax.

        Returns the `items` payload from `/search/repositories` as a typed contract.
        """

        response = await self._request(
            "/search/repositories",
            params={
                "q": query,
                "page": page,
                "per_page": per_page,
                "sort": sort,
                "order": order,
            },
            etag=etag,
            accept=self.ACCEPT_JSON,
        )
        if response.state != FetchState.OK:
            return FetchResult(
                state=response.state,
                data=None,
                etag=response.etag,
                status_code=response.status_code,
                error=response.error,
            )

        payload = response.data if isinstance(response.data, dict) else {}
        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        if not items:
            return FetchResult(
                state=FetchState.EMPTY,
                data=[],
                etag=response.etag,
                status_code=response.status_code,
            )

        return FetchResult(
            state=FetchState.OK,
            data=items,
            etag=response.etag,
            status_code=response.status_code,
        )

    async def list_releases(
        self,
        owner: str,
        repo: str,
        *,
        per_page: int = 100,
        etag: Optional[str] = None,
    ) -> ReleaseContract:
        path = f"/repos/{owner}/{repo}/releases"
        return await self._fetch_json_contract(path, params={"per_page": per_page}, etag=etag)

    async def list_tags(
        self,
        owner: str,
        repo: str,
        *,
        per_page: int = 100,
        etag: Optional[str] = None,
    ) -> TagContract:
        path = f"/repos/{owner}/{repo}/tags"
        return await self._fetch_json_contract(path, params={"per_page": per_page}, etag=etag)

    async def list_stargazers(
        self,
        owner: str,
        repo: str,
        *,
        page: int = 1,
        per_page: int = 100,
        etag: Optional[str] = None,
    ) -> StargazerContract:
        path = f"/repos/{owner}/{repo}/stargazers"
        return await self._fetch_json_contract(
            path,
            params={"page": page, "per_page": per_page},
            etag=etag,
            accept=self.ACCEPT_STARGAZERS,
        )

    async def get_content(self, owner: str, repo: str, path: str, etag: Optional[str] = None) -> ContentContract:
        response = await self._request(
            f"/repos/{owner}/{repo}/contents/{path}",
            etag=etag,
            accept=self.ACCEPT_JSON,
        )
        if response.state != FetchState.OK:
            return FetchResult(
                state=response.state,
                data=None,
                etag=response.etag,
                status_code=response.status_code,
                error=response.error,
            )

        payload = response.data if isinstance(response.data, dict) else {}
        encoded = payload.get("content") if isinstance(payload.get("content"), str) else ""
        encoding = payload.get("encoding") if isinstance(payload.get("encoding"), str) else ""

        if not encoded:
            return FetchResult(state=FetchState.EMPTY, data="", etag=response.etag, status_code=response.status_code)

        if encoding == "base64":
            try:
                decoded = base64.b64decode(encoded).decode("utf-8", errors="replace")
            except Exception as exc:
                return FetchResult(
                    state=FetchState.FAILED,
                    error=f"Failed to decode base64 content: {exc}",
                    etag=response.etag,
                    status_code=response.status_code,
                )
        else:
            decoded = encoded

        if not decoded.strip():
            return FetchResult(state=FetchState.EMPTY, data=decoded, etag=response.etag, status_code=response.status_code)

        return FetchResult(state=FetchState.OK, data=decoded, etag=response.etag, status_code=response.status_code)

    async def _fetch_json_contract(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        etag: Optional[str] = None,
        accept: Optional[str] = None,
    ) -> FetchResult[Any]:
        response = await self._request(path, params=params, etag=etag, accept=accept)
        if response.state != FetchState.OK:
            return response

        payload = response.data
        if payload is None:
            return FetchResult(
                state=FetchState.EMPTY,
                data=payload,
                etag=response.etag,
                status_code=response.status_code,
            )

        if isinstance(payload, (list, dict, str)) and len(payload) == 0:
            return FetchResult(
                state=FetchState.EMPTY,
                data=payload,
                etag=response.etag,
                status_code=response.status_code,
            )

        return response

    async def _request(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        etag: Optional[str] = None,
        accept: Optional[str] = None,
    ) -> FetchResult[Any]:
        client = await self._ensure_client()
        headers = self._build_headers(etag=etag, accept=accept)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=self._backoff_base_seconds, max=self._backoff_max_seconds),
                retry=retry_if_exception_type(_RateLimitRetryableError),
                reraise=True,
            ):
                with attempt:
                    response = await client.get(path, params=params, headers=headers)

                    if response.status_code == 304:
                        return FetchResult(
                            state=FetchState.UNCHANGED,
                            data=None,
                            etag=response.headers.get("etag", etag),
                            status_code=response.status_code,
                        )

                    if response.status_code in (403, 429):
                        wait_seconds = self._compute_rate_limit_wait(response.headers)
                        logger.warning(
                            "GitHub API rate limit encountered",
                            extra=sanitize_log_extra(
                                path=path,
                                params=params,
                                status_code=response.status_code,
                                retry_after_seconds=wait_seconds,
                            ),
                        )
                        if wait_seconds > 0:
                            await asyncio.sleep(wait_seconds)
                        raise _RateLimitRetryableError(
                            f"GitHub rate limit encountered ({response.status_code})"
                        )

                    response.raise_for_status()
                    return FetchResult(
                        state=FetchState.OK,
                        data=response.json(),
                        etag=response.headers.get("etag", etag),
                        status_code=response.status_code,
                    )
        except _RateLimitRetryableError as exc:
            logger.warning(
                "GitHub request failed after rate-limit retries",
                extra=sanitize_log_extra(path=path, params=params, error=str(exc), status_code=429),
            )
            return FetchResult(
                state=FetchState.FAILED,
                error=str(exc),
                etag=etag,
                status_code=429,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "GitHub request failed",
                extra=sanitize_log_extra(
                    path=path,
                    params=params,
                    error=str(exc),
                    status_code=getattr(getattr(exc, "response", None), "status_code", None),
                ),
            )
            return FetchResult(
                state=FetchState.FAILED,
                error=str(exc),
                etag=etag,
                status_code=getattr(getattr(exc, "response", None), "status_code", None),
            )

        return FetchResult(state=FetchState.FAILED, error="Unknown GitHub request failure", etag=etag)

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client:
            return self._client

        headers = {
            "Accept": self.ACCEPT_JSON,
            "User-Agent": settings.USER_AGENT,
            "X-GitHub-Api-Version": self.API_VERSION,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )
        return self._client

    def _build_headers(self, *, etag: Optional[str], accept: Optional[str]) -> dict[str, str]:
        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if accept:
            headers["Accept"] = accept
        return headers

    def _compute_rate_limit_wait(self, headers: httpx.Headers) -> float:
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass

        reset_raw = headers.get("x-ratelimit-reset")
        if reset_raw is not None:
            try:
                reset_epoch = int(reset_raw)
                wait_seconds = reset_epoch - int(time.time()) + self._rate_limit_buffer_seconds
                return float(max(wait_seconds, 0))
            except ValueError:
                pass

        return self._backoff_base_seconds
