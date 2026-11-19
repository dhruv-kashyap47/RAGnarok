"""Tests for the request body size limit (unit + endpoint level)."""

import pytest
from fastapi import HTTPException

from app.core.middleware import BodySizeLimitMiddleware


def _scope(content_length: int | None) -> dict:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {"type": "http", "headers": headers, "method": "POST", "path": "/x"}


async def test_declared_oversize_body_is_refused_on_first_read():
    app_reached_after_read = []

    async def downstream_app(scope, receive, send):
        await receive()  # simulates the route reading the body
        app_reached_after_read.append(True)

    middleware = BodySizeLimitMiddleware(downstream_app, max_bytes=100)

    async def receive():
        return {"type": "http.request", "body": b"x"}

    async def send(message):
        pass

    with pytest.raises(HTTPException) as exc:
        await middleware(_scope(content_length=10_000), receive, send)
    assert exc.value.status_code == 413

    # The body was refused before the route could process it.
    assert app_reached_after_read == []


async def test_lying_content_length_is_caught_while_streaming():
    """Chunked/lying uploads without an honest Content-Length still abort."""

    async def downstream_app(scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            if not message.get("body"):
                break

    middleware = BodySizeLimitMiddleware(downstream_app, max_bytes=100)

    sent = 0

    async def receive():
        nonlocal sent
        sent += 1
        return {"type": "http.request", "body": b"x" * 60}

    async def send(message):
        pass

    with pytest.raises(HTTPException) as exc:
        await middleware(_scope(content_length=None), receive, send)
    assert exc.value.status_code == 413

    # 60 + 60 > 100: aborted on the second frame, not after buffering forever.
    assert sent == 2


async def test_bodies_within_limit_pass_through():
    read = []

    async def downstream_app(scope, receive, send):
        message = await receive()
        read.append(message.get("body"))

    middleware = BodySizeLimitMiddleware(downstream_app, max_bytes=100)

    async def receive():
        return {"type": "http.request", "body": b"small"}

    async def send(message):
        pass

    await middleware(_scope(content_length=5), receive, send)
    assert read == [b"small"]


async def test_non_http_scopes_are_untouched():
    called = []

    async def downstream_app(scope, receive, send):
        called.append(scope["type"])

    middleware = BodySizeLimitMiddleware(downstream_app, max_bytes=100)
    await middleware({"type": "lifespan"}, None, None)
    assert called == ["lifespan"]


def test_oversize_json_endpoint_returns_413(client):
    """End-to-end: oversized request bodies are refused with the API shape."""
    payload = {"email": "a@b.com", "password": "x" * (1024 * 1024 + 100)}
    response = client.post("/login", json=payload)
    assert response.status_code == 413
    body = response.json()
    assert body["success"] is False
    assert "too large" in body["message"].lower()
