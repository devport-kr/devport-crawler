"""Shared crawler completion webhook dispatcher.

Sends an HMAC-signed POST to the Spring API so it can invalidate caches for
a given scope (e.g. GIT_REPO) after a crawl finishes. Matches the payload
shape expected by CrawlerJobCompletedRequest on the api-dev side.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import random
from typing import Any

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


async def dispatch_completion_webhook(
    *,
    scope: str,
    job_id: str,
    completed_at: str,
) -> dict[str, Any] | None:
    """Dispatch a crawler-completion webhook with retry/backoff.

    Returns None when webhook config is missing, otherwise a status dict
    suitable for attaching to crawl stats.
    """
    webhook_url = str(getattr(settings, "CRAWLER_WEBHOOK_URL", "") or "").strip()
    webhook_secret = str(getattr(settings, "CRAWLER_WEBHOOK_SECRET", "") or "").strip()
    if not webhook_url or not webhook_secret:
        return None

    payload = {
        "job_id": job_id,
        "scope": scope,
        "completed_at": completed_at,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    signature = hmac.new(
        webhook_secret.encode("utf-8"),
        payload_json.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    signed_payload = {**payload, "signature": signature}

    max_retries = max(int(getattr(settings, "CRAWLER_WEBHOOK_MAX_RETRIES", 3) or 3), 1)
    timeout_seconds = float(getattr(settings, "CRAWLER_WEBHOOK_TIMEOUT_SECONDS", 10.0) or 10.0)

    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(webhook_url, json=signed_payload)
            if response.status_code < 400:
                return {
                    "sent": True,
                    "attempts": attempt,
                    "status_code": response.status_code,
                }
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < max_retries:
            await asyncio.sleep(_retry_delay_seconds(attempt))

    logger.warning(
        "Crawler completion webhook delivery failed: scope=%s, error=%s, retries=%s",
        scope,
        last_error,
        max_retries,
    )
    return {
        "sent": False,
        "attempts": max_retries,
        "error": last_error or "unknown error",
    }


def _retry_delay_seconds(attempt: int) -> float:
    base = min(0.1 * (2 ** max(attempt - 1, 0)), 2.0)
    jitter = random.uniform(0.75, 1.25)
    return base * jitter
