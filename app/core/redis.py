import redis
from redis.exceptions import RedisError

from app.core.config import REDIS_URL
from app.core.logger import logger

_pool = None
_client = None


def _get_client():
    """Lazy-init Redis with connection pooling. Returns None if unavailable."""
    global _pool, _client

    if _client is not None:
        return _client

    if not REDIS_URL:
        return None

    try:
        _pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            decode_responses=True,
            max_connections=10,
            socket_connect_timeout=2,
            socket_timeout=1,
            retry_on_timeout=True,
        )
        _client = redis.Redis(connection_pool=_pool)
        _client.ping()
        logger.info("Redis connected: %s", REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL)
        return _client
    except Exception as exc:
        logger.warning("Redis unavailable, running without cache: %s", exc)
        _client = None
        _pool = None
        return None


def safe_cache_get(key: str):
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get(key)
    except RedisError as exc:
        logger.warning("Redis GET failed key=%s: %s", key, exc)
        return None


def safe_cache_set(key: str, value: str, ex: int | None = None) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.set(key, value, ex=ex)
        return True
    except RedisError as exc:
        logger.warning("Redis SET failed key=%s: %s", key, exc)
        return False


def safe_cache_delete(key: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.delete(key)
        return True
    except RedisError as exc:
        logger.warning("Redis DEL failed key=%s: %s", key, exc)
        return False
