"""Unit tests for the Groq-only chat client: config, SSE parsing, errors."""

import json

import httpx
import pytest
from fastapi import HTTPException

from app.services import llm_service

SSE_OK = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


@pytest.fixture()
def groq_key(monkeypatch):
    monkeypatch.setattr(llm_service, "GROQ_API_KEY", "gsk_test_key")


@pytest.fixture(autouse=True)
def _reset_model_state(monkeypatch):
    """Keep the sticky/discovered model cache from leaking between tests."""
    monkeypatch.setattr(llm_service, "_active_model", None)
    monkeypatch.setattr(llm_service, "_discovered_models", None)
    monkeypatch.setattr(llm_service, "GROQ_CHAT_MODEL", "primary-model")
    monkeypatch.setattr(llm_service, "GROQ_FALLBACK_MODELS", ["fallback-model"])


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
        return httpx.Response(
            200, text=sse, headers={"content-type": "text/event-stream"}
        )

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


async def test_stream_falls_back_when_primary_model_is_retired(groq_key, monkeypatch):
    """Groq retires models (e.g. llama-3.1-8b-instant); a 404 on the primary
    must transparently move to the fallback and stick with it afterwards."""
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        requested_models.append(model)
        if model == "primary-model":
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": "The model `primary-model` does not exist "
                        "or you do not have access to it.",
                        "code": "model_not_found",
                    }
                },
            )
        return httpx.Response(
            200, text=SSE_OK, headers={"content-type": "text/event-stream"}
        )

    test_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_service, "_get_client", lambda: test_client)

    chunks = [c async for c in llm_service.stream_generate_answer("prompt")]
    assert chunks == ["ok"]
    assert requested_models == ["primary-model", "fallback-model"]
    # Capabilities and later requests now use the model that actually works.
    assert llm_service.get_llm_model() == "fallback-model"

    again = [c async for c in llm_service.stream_generate_answer("again")]
    assert again == ["ok"]
    assert requested_models[-1] == "fallback-model"
    await test_client.aclose()


async def test_discovers_a_chat_model_when_all_configured_models_are_gone(
    groq_key, monkeypatch
):
    """If every configured model is retired, ask Groq's /models endpoint for a
    usable chat model instead of failing until someone edits an env var."""
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "whisper-large-v3"},
                        {"id": "openai/gpt-oss-20b"},
                        {"id": "nomic-embed-text"},
                    ]
                },
            )

        model = json.loads(request.content)["model"]
        requested_models.append(model)
        if model == "openai/gpt-oss-20b":
            return httpx.Response(
                200, text=SSE_OK, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": f"The model `{model}` does not exist.",
                    "code": "model_not_found",
                }
            },
        )

    test_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_service, "_get_client", lambda: test_client)

    chunks = [c async for c in llm_service.stream_generate_answer("prompt")]
    assert chunks == ["ok"]
    # Configured candidates first, then the discovered chat model. Whisper and
    # embedding models are never sent to /chat/completions.
    assert requested_models == [
        "primary-model",
        "fallback-model",
        "openai/gpt-oss-20b",
    ]
    assert llm_service.get_llm_model() == "openai/gpt-oss-20b"
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
