"""Invocation deadline shared by crawlers and the summarizer.

Lambda kills the process at its timeout, so long phases check how much time is
left before starting work they could not finish (a Playwright render, an LLM
translation). Outside Lambda no deadline is set and everything runs.
"""

import math
import time
from typing import Optional

_deadline: Optional[float] = None


def set_deadline(seconds_from_now: Optional[float]) -> None:
    global _deadline
    _deadline = None if seconds_from_now is None else time.monotonic() + seconds_from_now


def remaining() -> float:
    """Seconds left before the deadline (infinite when none is set)."""
    if _deadline is None:
        return math.inf
    return _deadline - time.monotonic()
