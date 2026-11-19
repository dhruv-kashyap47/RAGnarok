import time

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import REDIS_URL
from app.core.logger import logger

_client: aioredis.Redis | None = None
# After a failed connection attempt, do not retry for this many seconds so a
# down Redis does not add a connect timeout to every single request.
_RETRY_BACKOFF_SECONDS = 30.0
_next_retry_at = 0.0


async def _get_client() -> aioredis.Redis | None:
    """Lazy-init Redis with connection pooling. Returns None if unavailable."""
    global _client, _next_retry_at

    if _client is not None:
        return _client

    if not REDIS_URL:
        return None

    now = time.monotonic()
    if now < _next_retry_at:
        return None

    try:
        client = aioredis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            max_connections=10,
            socket_connect_timeout=2,
            socket_timeout=1,
        )
        await client.ping()
        _client = client
        logger.info(
            "Redis connected: %s",
            REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL,
        )
        return _client
    except Exception as exc:  # noqa: BLE001 — any failure means "no cache"
        logger.warning("Redis unavailable, running without cache: %s", exc)
        _client = None
        _next_retry_at = now + _RETRY_BACKOFF_SECONDS
        return None


async def get_redis_client() -> aioredis.Redis | None:
    """Public accessor: shared async client, or None while Redis is down."""
    return await _get_client()


async def close_redis() -> None:
    """Close the shared Redis client on application shutdown."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as exc:  # noqa: BLE001 — best-effort shutdown
            logger.debug("Redis close failed: %s", exc)
        _client = None


async def safe_cache_get(key: str):
    client = await _get_client()
    if client is None:
        return None
    try:
        return await client.get(key)
    except RedisError as exc:
        logger.warning("Redis GET failed key=%s: %s", key, exc)
        return None


async def safe_cache_set(key: str, value: str, ex: int | None = None) -> bool:
    client = await _get_client()
    if client is None:
        return False
    try:
        await client.set(key, value, ex=ex)
        return True
    except RedisError as exc:
        logger.warning("Redis SET failed key=%s: %s", key, exc)
        return False


async def safe_cache_delete(key: str) -> bool:
    client = await _get_client()
    if client is None:
        return False
    try:
        await client.delete(key)
        return True
    except RedisError as exc:
        logger.warning("Redis DEL failed key=%s: %s", key, exc)
        return False
