"""
Sliding-window rate limiting.

The authoritative store is Redis (a Lua script over a ZSET), so limits are
global across uvicorn workers and survive worker restarts. When Redis is
unavailable the limiter degrades to the previous in-process table: limits
then apply per worker (an intentional availability-over-strictness
trade-off) instead of failing every request.

Keys are prefixed with ``rl:`` in Redis so tests can flush them safely.
"""

from __future__ import annotations

import time
import uuid

from fastapi import HTTPException, Request
from redis.exceptions import RedisError

from app.core.config import TRUST_PROXY_HEADERS
from app.core.logger import logger
from app.core.redis import get_redis_client

_DEFAULT_WINDOW_SECONDS = 60.0
# Prune the hit table once it grows past this many distinct keys so a
# key-spraying client cannot grow memory without bound.
_MAX_KEYS = 10_000
_KEY_PREFIX = "rl:"

_hits: dict[str, list[float]] = {}

# Sliding window over a ZSET: drop expired members, count the rest, and only
# record the hit when under the limit. One atomic round-trip per request.
_LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
if redis.call('ZCARD', key) >= limit then
  return 0
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, math.ceil(window * 1000))
return 1
"""


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
    """Clear all in-process hit counters (used by tests)."""
    _hits.clear()


def check_rate_limit(key: str, limit: int, window_seconds: float = _DEFAULT_WINDOW_SECONDS) -> bool:
    """In-memory sliding window: True if *key* is within *limit* hits/window.

    This is the fallback path when Redis is unavailable (and the unit-test
    surface for window behavior; it is ``_Clock``-patchable).
    """
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


async def check_rate_limit_redis(
    key: str, limit: int, window_seconds: float = _DEFAULT_WINDOW_SECONDS
) -> bool | None:
    """Redis-backed sliding window.

    Returns True (allowed) / False (limited), or None when Redis is
    unavailable so the caller can fall back to the in-process table.
    """
    if limit <= 0:
        return True

    client = await get_redis_client()
    if client is None:
        return None

    try:
        # Wall-clock seconds: buckets are shared across workers, so a common
        # time base (not per-process monotonic) is required.
        now = time.time()
        allowed = await client.eval(
            _LUA_SLIDING_WINDOW,
            1,
            _KEY_PREFIX + key,
            window_seconds,
            limit,
            now,
            uuid.uuid4().hex,
        )
        return int(allowed) == 1
    except (RedisError, ValueError, TypeError) as exc:
        logger.warning("Redis rate-limit check failed key=%s: %s", key, exc)
        return None


async def check_rate_limit_any(
    key: str, limit: int, window_seconds: float = _DEFAULT_WINDOW_SECONDS
) -> bool:
    """Redis-backed check with in-memory fallback when Redis is down."""
    result = await check_rate_limit_redis(key, limit, window_seconds)
    if result is None:
        return check_rate_limit(key, limit, window_seconds)
    return result


def _prune(now: float, window_seconds: float) -> None:
    cutoff = now - window_seconds
    stale = [key for key, hits in _hits.items() if not hits or hits[-1] <= cutoff]
    for key in stale:
        _hits.pop(key, None)
    if len(_hits) >= _MAX_KEYS:
        # Still full of fresh keys: hard-reset to stay memory-bounded.
        logger.warning("Rate-limit table full; resetting counters")
        _hits.clear()


async def enforce_rate_limit(
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
    if not await check_rate_limit_any(bucket, limit, window_seconds):
        retry_after = max(1, int(window_seconds))
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again later.",
            headers={"Retry-After": str(retry_after)},
        )


async def enforce_identity_rate_limit(
    request: Request,
    identity: dict,
    kind: str,
    limit: int,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
) -> None:
    """Throttle by identity, plus by IP for guests.

    Guests could otherwise mint unlimited fresh sessions (one bucket per
    guest user id) to bypass per-identity limits, so guest traffic is also
    counted against the caller's IP.
    """
    await enforce_rate_limit(request, f"{kind}:{identity['user_id']}", limit, window_seconds)
    if identity.get("role") == "guest":
        await enforce_rate_limit(
            request, f"{kind}-ip:{client_key(request)}", limit, window_seconds
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
