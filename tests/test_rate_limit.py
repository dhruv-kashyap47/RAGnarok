"""Tests for the sliding-window rate limiter: unit behavior + endpoint 429s."""

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


# --- Unit: sliding window --------------------------------------------------


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


def test_enforce_rate_limit_raises_429_with_retry_after():
    rate_limit.reset_rate_limits()
    req = _request()
    rate_limit.enforce_rate_limit(req, "unit:enforce", 1)

    with pytest.raises(HTTPException) as exc:
        rate_limit.enforce_rate_limit(req, "unit:enforce", 1)
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


def test_reset_rate_limits_clears_buckets():
    rate_limit.reset_rate_limits()
    assert rate_limit.check_rate_limit("unit:reset", 1) is True
    assert rate_limit.check_rate_limit("unit:reset", 1) is False
    rate_limit.reset_rate_limits()
    assert rate_limit.check_rate_limit("unit:reset", 1) is True
