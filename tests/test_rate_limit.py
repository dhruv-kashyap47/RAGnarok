"""Tests for the sliding-window rate limiter: unit behavior + endpoint 429s.

Two backends are covered: the Redis Lua script (authoritative, global across
workers) and the in-process fallback used while Redis is unavailable.
"""

import asyncio
import time as real_time

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.core import rate_limit


def _request(peer: str = "9.9.9.9", headers: dict[str, str] | None = None) -> Request:
    scope = {
        "type": "http",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "client": (peer, 12345),
        "method": "GET",
        "path": "/x",
        "query_string": b"",
    }
    return Request(scope)


async def _no_redis():
    """Force the in-memory fallback path (deterministic unit tests)."""
    return


# --- Unit: sliding window (in-memory fallback) -----------------------------


def test_allows_up_to_limit_then_blocks():
    rate_limit.reset_rate_limits()
    key = "unit:window"
    assert [rate_limit.check_rate_limit(key, 3) for _ in range(4)] == [
        True,
        True,
        True,
        False,
    ]


class _Clock:
    """Controllable stand-in for the ``time`` module inside rate_limit."""

    def __init__(self, start: float):
        self.now = start

    def monotonic(self) -> float:
        return self.now


def test_window_expiry_frees_the_bucket(monkeypatch):
    clock = _Clock(real_time.monotonic())
    monkeypatch.setattr(rate_limit, "time", clock)

    key = "unit:expiry"
    assert rate_limit.check_rate_limit(key, 1) is True
    assert rate_limit.check_rate_limit(key, 1) is False

    # Still blocked inside the window.
    clock.now += 30
    assert rate_limit.check_rate_limit(key, 1) is False

    # Free again after the 60s window elapses.
    clock.now += 31
    assert rate_limit.check_rate_limit(key, 1) is True


def test_non_positive_limit_disables_throttling():
    rate_limit.reset_rate_limits()
    key = "unit:disabled"
    assert rate_limit.check_rate_limit(key, 0) is True
    assert rate_limit.check_rate_limit(key, -5) is True


async def test_enforce_rate_limit_raises_429_with_retry_after(monkeypatch):
    monkeypatch.setattr(rate_limit, "get_redis_client", _no_redis)
    rate_limit.reset_rate_limits()
    req = _request()
    await rate_limit.enforce_rate_limit(req, "unit:enforce", 1)

    with pytest.raises(HTTPException) as exc:
        await rate_limit.enforce_rate_limit(req, "unit:enforce", 1)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "60"
    assert "Too many requests" in exc.value.detail


def test_hit_table_stays_memory_bounded(monkeypatch):
    """A key-spraying client must not grow the table without bound: stale
    entries are pruned when the table is full."""
    rate_limit.reset_rate_limits()
    clock = _Clock(real_time.monotonic())
    monkeypatch.setattr(rate_limit, "time", clock)

    for i in range(rate_limit._MAX_KEYS):
        rate_limit._hits[f"stale-{i}"] = [clock.now - 999]

    assert rate_limit.check_rate_limit("fresh-key", 5) is True
    assert len(rate_limit._hits) <= 1  # stale keys were pruned


# --- Unit: Redis Lua sliding window ----------------------------------------


async def test_redis_sliding_window_blocks_and_expires(infra_ready, monkeypatch):
    """The Lua script must count, block, and expire hits in Redis."""
    import redis.asyncio as aioredis

    from tests.conftest import TEST_REDIS_URL

    rclient = aioredis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)

    async def use_test_client():
        return rclient

    monkeypatch.setattr(rate_limit, "get_redis_client", use_test_client)

    key = "unit:redis-window"
    try:
        assert await rate_limit.check_rate_limit_redis(key, 2) is True
        assert await rate_limit.check_rate_limit_redis(key, 2) is True
        assert await rate_limit.check_rate_limit_redis(key, 2) is False

        # Expiry: with a tiny window the single hit ages out quickly.
        assert await rate_limit.check_rate_limit_redis("unit:redis-expiry", 1, 0.05) is True
        assert await rate_limit.check_rate_limit_redis("unit:redis-expiry", 1, 0.05) is False
        await asyncio.sleep(0.06)
        assert await rate_limit.check_rate_limit_redis("unit:redis-expiry", 1, 0.05) is True
    finally:
        await rclient.aclose()


async def test_redis_unavailable_falls_back_to_memory(monkeypatch):
    monkeypatch.setattr(rate_limit, "get_redis_client", _no_redis)
    rate_limit.reset_rate_limits()
    key = "unit:fallback"
    assert await rate_limit.check_rate_limit_any(key, 1) is True
    assert await rate_limit.check_rate_limit_any(key, 1) is False
    # The hit was recorded in-process, not lost.
    assert rate_limit._hits


async def test_non_positive_limit_never_touches_redis(monkeypatch):
    async def boom():
        raise AssertionError("Redis must not be consulted for disabled limits")

    monkeypatch.setattr(rate_limit, "get_redis_client", boom)
    assert await rate_limit.check_rate_limit_redis("unit:noop", 0) is True


async def test_enforce_identity_rate_limit_guests_also_count_ip(monkeypatch):
    """A guest could mint fresh sessions to reset its identity bucket; the
    extra IP bucket closes that bypass."""
    monkeypatch.setattr(rate_limit, "get_redis_client", _no_redis)
    rate_limit.reset_rate_limits()
    req = _request(peer="5.5.5.5")
    guest1 = {"user_id": "guest-1", "role": "guest"}
    guest2 = {"user_id": "guest-2", "role": "guest"}
    user = {"user_id": "user-1", "role": "user"}

    await rate_limit.enforce_identity_rate_limit(req, guest1, "ask", 1)
    with pytest.raises(HTTPException):
        await rate_limit.enforce_identity_rate_limit(req, guest1, "ask", 1)
    # New guest id, same IP: blocked by the shared guest IP bucket.
    with pytest.raises(HTTPException):
        await rate_limit.enforce_identity_rate_limit(req, guest2, "ask", 1)
    # Registered users are only throttled per identity.
    await rate_limit.enforce_identity_rate_limit(req, user, "ask", 1)


# --- Unit: bucket keys -----------------------------------------------------


def test_client_key_uses_peer_by_default():
    assert rate_limit.client_key(_request(peer="1.2.3.4")) == "1.2.3.4"


def test_client_key_ignores_forwarded_headers_when_untrusted(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUST_PROXY_HEADERS", False)
    req = _request(peer="1.2.3.4", headers={"X-Forwarded-For": "6.6.6.6"})
    assert rate_limit.client_key(req) == "1.2.3.4"


def test_client_key_honours_first_forwarded_hop_when_trusted(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUST_PROXY_HEADERS", True)
    req = _request(peer="10.0.0.1", headers={"X-Forwarded-For": "6.6.6.6, 10.0.0.1"})
    assert rate_limit.client_key(req) == "6.6.6.6"


def test_auth_bucket_isolates_emails_and_normalizes_case():
    req = _request(peer="1.2.3.4")
    a1 = rate_limit.auth_bucket(req, "login", "User@Example.COM")
    a2 = rate_limit.auth_bucket(req, "login", "user@example.com")
    other = rate_limit.auth_bucket(req, "login", "other@example.com")
    no_email = rate_limit.auth_bucket(req, "register")
    assert a1 == a2  # normalized
    assert a1 != other  # separate buckets per account
    assert no_email == "register:1.2.3.4"


# --- Endpoint level --------------------------------------------------------


def test_login_rate_limit_returns_429_shape(client, monkeypatch):
    import app.api.user as api_user

    monkeypatch.setattr(api_user, "AUTH_RATE_LIMIT_PER_MINUTE", 2)
    payload = {"email": "brute@example.com", "password": "nope-nope"}

    assert client.post("/login", json=payload).status_code == 400  # bad creds, counted
    assert client.post("/login", json=payload).status_code == 400

    response = client.post("/login", json=payload)
    assert response.status_code == 429
    body = response.json()
    assert body["success"] is False
    assert "Too many requests" in body["message"]
    assert response.headers["Retry-After"] == "60"


def test_login_429_bucket_is_recorded_in_redis(client, monkeypatch):
    """Regression: limits must live in Redis (global across workers), not in
    the per-process table — otherwise WEB_CONCURRENCY doubles the limit."""
    import redis as sync_redis

    import app.api.user as api_user
    from tests.conftest import TEST_REDIS_URL

    monkeypatch.setattr(api_user, "AUTH_RATE_LIMIT_PER_MINUTE", 2)
    payload = {"email": "redis-brute@example.com", "password": "nope-nope"}

    for _ in range(3):
        client.post("/login", json=payload)

    # The in-process table never saw a hit: the Lua script handled them all.
    assert rate_limit._hits == {}

    r = sync_redis.Redis.from_url(
        TEST_REDIS_URL, socket_connect_timeout=2, socket_timeout=2
    )
    try:
        keys = list(r.scan_iter("rl:login:*"))
    finally:
        r.close()
    assert keys, "expected the login bucket to exist in Redis"


def test_login_bucket_is_per_email_not_per_ip(client, monkeypatch):
    """One account being brute-forced must not lock out other accounts that
    share the same egress IP."""
    import uuid

    import app.api.user as api_user

    monkeypatch.setattr(api_user, "AUTH_RATE_LIMIT_PER_MINUTE", 2)

    victim = {"email": "victim@example.com", "password": "wrong-pass"}
    assert client.post("/login", json=victim).status_code == 400
    assert client.post("/login", json=victim).status_code == 400
    assert client.post("/login", json=victim).status_code == 429

    # Different email, same IP: still allowed.
    other_email = f"other-{uuid.uuid4().hex[:8]}@example.com"
    assert client.post("/users", json={"email": other_email, "password": "secret123"}).status_code == 200
    assert (
        client.post("/login", json={"email": other_email, "password": "secret123"}).status_code
        == 200
    )


def test_register_rate_limit(client, monkeypatch):
    import uuid

    import app.api.user as api_user

    monkeypatch.setattr(api_user, "REGISTER_RATE_LIMIT_PER_MINUTE", 2)

    for _ in range(2):
        email = f"bulk-{uuid.uuid4().hex[:8]}@example.com"
        assert client.post("/users", json={"email": email, "password": "secret123"}).status_code == 200

    email = f"bulk-{uuid.uuid4().hex[:8]}@example.com"
    response = client.post("/users", json={"email": email, "password": "secret123"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_guest_session_rate_limit(client, monkeypatch):
    """Minting guest sessions must be throttled per IP: otherwise a client
    could generate unlimited fresh guest identities."""
    import app.api.user as api_user

    monkeypatch.setattr(api_user, "GUEST_SESSION_RATE_LIMIT_PER_MINUTE", 2)

    assert client.post("/guest-session").status_code == 200
    assert client.post("/guest-session").status_code == 200
    response = client.post("/guest-session")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_ask_rate_limit_per_user(client, account, mock_ask, monkeypatch):
    import app.api.user as api_user

    monkeypatch.setattr(api_user, "ASK_RATE_LIMIT_PER_MINUTE", 1)

    first = client.get(
        "/ask", params={"question": "q1"}, headers=account["headers"]
    )
    assert first.status_code == 200

    second = client.get(
        "/ask", params={"question": "q2"}, headers=account["headers"]
    )
    assert second.status_code == 429
    assert "Too many requests" in second.json()["message"]


def test_ask_rate_limit_isolated_between_users(
    client, account, mock_ask, monkeypatch
):
    import uuid

    import app.api.user as api_user

    monkeypatch.setattr(api_user, "ASK_RATE_LIMIT_PER_MINUTE", 1)

    exhausted = client.get(
        "/ask", params={"question": "q1"}, headers=account["headers"]
    )
    assert exhausted.status_code == 200
    assert (
        client.get("/ask", params={"question": "q2"}, headers=account["headers"]).status_code
        == 429
    )

    other_email = f"other-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/users", json={"email": other_email, "password": "secret123"})
    other_token = client.post(
        "/login", json={"email": other_email, "password": "secret123"}
    ).json()["access_token"]
    other = {"Authorization": f"Bearer {other_token}"}

    assert client.get("/ask", params={"question": "q1"}, headers=other).status_code == 200


def test_guest_ask_rate_limit_blocks_fresh_guests_from_same_ip(
    client, guest_account, mock_ask, monkeypatch
):
    """A new guest session from the same IP must not reset an exhausted
    guest's ask budget (the IP bucket)."""
    import app.api.user as api_user

    monkeypatch.setattr(api_user, "ASK_RATE_LIMIT_PER_MINUTE", 2)

    def ask(headers):
        return client.get("/ask", params={"question": "q"}, headers=headers)

    assert ask(guest_account["headers"]).status_code == 200
    assert ask(guest_account["headers"]).status_code == 200
    assert ask(guest_account["headers"]).status_code == 429

    fresh_guest = client.post("/guest-session").json()
    fresh_headers = {"Authorization": f"Bearer {fresh_guest['access_token']}"}
    # Fresh identity bucket, but the shared guest IP bucket is already full.
    assert ask(fresh_headers).status_code == 429


def test_upload_rate_limit(client, account, mock_embeddings, monkeypatch):
    import app.api.user as api_user
    from tests.conftest import make_pdf

    monkeypatch.setattr(api_user, "UPLOAD_RATE_LIMIT_PER_MINUTE", 1)

    def upload():
        return client.post(
            "/upload-pdf",
            headers=account["headers"],
            files={"file": ("a.pdf", make_pdf("some text for the document"), "application/pdf")},
        )

    assert upload().status_code == 200
    second = upload()
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"


def test_reset_rate_limits_clears_buckets(monkeypatch):
    monkeypatch.setattr(rate_limit, "get_redis_client", _no_redis)
    rate_limit.reset_rate_limits()
    assert rate_limit.check_rate_limit("unit:reset", 1) is True
    assert rate_limit.check_rate_limit("unit:reset", 1) is False
    rate_limit.reset_rate_limits()
    assert rate_limit.check_rate_limit("unit:reset", 1) is True
