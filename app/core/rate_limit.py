"""
Lightweight in-process sliding-window rate limiting.

Keys are scoped per worker process (uvicorn runs multiple workers, so the
effective global limit is approximately ``limit x workers``). This is an
intentional trade-off: no external dependency, works when Redis is down, and
never adds latency to the request path. Residual risk is documented in the
project report.
"""

from __future__ import annotations

import time

from fastapi import HTTPException, Request

from app.core.config import TRUST_PROXY_HEADERS
from app.core.logger import logger

_DEFAULT_WINDOW_SECONDS = 60.0
# Prune the hit table once it grows past this many distinct keys so a
# key-spraying client cannot grow memory without bound.
_MAX_KEYS = 10_000

_hits: dict[str, list[float]] = {}


def client_key(request: Request) -> str:
    """Best-effort client identifier for rate-limit bucketing."""
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # First hop is the original client when set by a trusted proxy.
            return forwarded.split(",")[0].strip() or _peer(request)
    return _peer(request)


def _peer(request: Request) -> str:
    client = request.client
    return client.host if client else "unknown"


def reset_rate_limits() -> None:
    """Clear all hit counters (used by tests)."""
    _hits.clear()


def check_rate_limit(key: str, limit: int, window_seconds: float = _DEFAULT_WINDOW_SECONDS) -> bool:
    """Return True if *key* is within *limit* hits per window, else False."""
    if limit <= 0:
        # Non-positive limits disable throttling for this bucket.
        return True

    now = time.monotonic()
    if len(_hits) >= _MAX_KEYS and key not in _hits:
        _prune(now, window_seconds)

    hits = _hits.setdefault(key, [])
    # Drop entries that fell out of the window.
    cutoff = now - window_seconds
    if hits:
        hits = [t for t in hits if t > cutoff]
        _hits[key] = hits

    if len(hits) >= limit:
        return False

    hits.append(now)
    return True


def _prune(now: float, window_seconds: float) -> None:
    cutoff = now - window_seconds
    stale = [key for key, hits in _hits.items() if not hits or hits[-1] <= cutoff]
    for key in stale:
        _hits.pop(key, None)
    if len(_hits) >= _MAX_KEYS:
        # Still full of fresh keys: hard-reset to stay memory-bounded.
        logger.warning("Rate-limit table full; resetting counters")
        _hits.clear()


def enforce_rate_limit(
    request: Request,
    bucket: str,
    limit: int,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
) -> None:
    """
    Raise HTTP 429 when the caller exceeds *limit* requests per window.

    ``bucket`` should already identify the subject (e.g. ``login:<ip>:<email>``
    or ``ask:<user_id>``).
    """
    if not check_rate_limit(bucket, limit, window_seconds):
        retry_after = max(1, int(window_seconds))
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def auth_bucket(request: Request, endpoint: str, email: str | None = None) -> str:
    """Build a rate-limit bucket for authentication endpoints."""
    peer = client_key(request)
    if email:
        # Keying on IP + email avoids locking out every user sharing one egress
        # IP (corporate NAT / reverse proxy) while still throttling targeted
        # brute-force attempts against a single account.
        return f"{endpoint}:{peer}:{email.strip().lower()}"
    return f"{endpoint}:{peer}"
