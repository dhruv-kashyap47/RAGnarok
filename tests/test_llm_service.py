"""Unit tests for the Groq-only chat client: config, SSE parsing, errors."""

import httpx
import pytest
from fastapi import HTTPException

from app.services import llm_service


@pytest.fixture()
def groq_key(monkeypatch):
    monkeypatch.setattr(llm_service, "GROQ_API_KEY", "gsk_test_key")


async def test_unconfigured_provider_raises(monkeypatch):
    monkeypatch.setattr(llm_service, "GROQ_API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        async for _ in llm_service.stream_generate_answer("hi"):
            pass
    assert exc.value.status_code == 500
    assert llm_service.get_llm_provider() == "unconfigured"


async def test_empty_prompt_is_rejected(groq_key):
    with pytest.raises(HTTPException) as exc:
        async for _ in llm_service.stream_generate_answer("   "):
            pass
    assert exc.value.status_code == 400


def test_provider_is_groq_when_configured(groq_key, monkeypatch):
    """Groq is the only provider: no fallback target exists."""
    assert llm_service.get_llm_provider() == "groq"
    assert llm_service.get_llm_model() == llm_service.GROQ_CHAT_MODEL
    monkeypatch.setattr(llm_service, "GROQ_API_KEY", None)
    assert llm_service.get_llm_provider() == "unconfigured"


async def test_groq_stream_parses_sse_and_skips_malformed_frames(groq_key, monkeypatch):
    """Malformed SSE frames must be skipped, not crash the stream."""
    sse = (
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        "data: {not-json\n\n"
        'data: {"choices":[]}\n\n'
        'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream" })

    test_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_service, "_get_client", lambda: test_client)

    chunks = [c async for c in llm_service.stream_generate_answer("prompt")]
    assert chunks == ["Hi", " there"]
    await test_client.aclose()


async def test_generate_answer_returns_stripped_text(groq_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "  hello  "}}]}
        )

    test_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_service, "_get_client", lambda: test_client)

    assert await llm_service.generate_answer("prompt") == "hello"
    await test_client.aclose()


async def test_generate_answer_transport_error_is_502(groq_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to https://internal-host")

    test_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await llm_service.generate_answer("prompt")
    assert exc.value.status_code == 502
    # No provider fallback exists anymore; internals must not leak either.
    assert "internal-host" not in str(exc.value.detail)
    await test_client.aclose()


async def test_generate_answer_http_error_is_502_with_provider_message(
    groq_key, monkeypatch
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    test_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await llm_service.generate_answer("prompt")
    assert exc.value.status_code == 502
    assert "rate limited" in str(exc.value.detail)
    await test_client.aclose()


async def test_stream_http_error_is_502(groq_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "server exploded"}})

    test_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        async for _ in llm_service.stream_generate_answer("prompt"):
            pass
    assert exc.value.status_code == 502
    assert "server exploded" in str(exc.value.detail)
    await test_client.aclose()


def test_sse_error_extraction():
    response = httpx.Response(
        429,
        json={"error": {"message": "rate limited"}},
        request=httpx.Request("POST", "https://example.test"),
    )
    assert llm_service._extract_error(response) == "rate limited"
    plain = httpx.Response(
        500, text="oops", request=httpx.Request("POST", "https://example.test")
    )
    assert llm_service._extract_error(plain) == "oops"
