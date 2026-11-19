"""Unit tests for provider selection and streaming fallback behavior."""

import httpx
import pytest
from fastapi import HTTPException

from app.services import llm_service


@pytest.fixture()
def providers(monkeypatch):
    monkeypatch.setattr(llm_service, "GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setattr(llm_service, "GEMINI_API_KEY", "ai_test_key")


async def test_unconfigured_provider_raises(monkeypatch):
    monkeypatch.setattr(llm_service, "GROQ_API_KEY", None)
    monkeypatch.setattr(llm_service, "GEMINI_API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        async for _ in llm_service.stream_generate_answer("hi"):
            pass
    assert exc.value.status_code == 500
    assert llm_service.get_llm_provider() == "unconfigured"


async def test_empty_prompt_is_rejected(providers):
    with pytest.raises(HTTPException) as exc:
        async for _ in llm_service.stream_generate_answer("   "):
            pass
    assert exc.value.status_code == 400


async def test_gemini_stream_used_when_only_gemini_configured(monkeypatch):
    monkeypatch.setattr(llm_service, "GROQ_API_KEY", None)
    monkeypatch.setattr(llm_service, "GEMINI_API_KEY", "ai_test_key")
    called = []

    async def fake_gemini(prompt):
        called.append(prompt)
        yield "g"

    monkeypatch.setattr(llm_service, "_stream_with_gemini", fake_gemini)
    chunks = [c async for c in llm_service.stream_generate_answer("hello")]
    assert chunks == ["g"]
    assert called == ["hello"]


async def test_groq_failure_before_first_chunk_falls_back(providers, monkeypatch):
    """A pre-stream failure must transparently retry via Gemini."""
    gemini_called = []

    async def failing_groq(prompt):
        raise HTTPException(status_code=502, detail="boom")
        yield  # pragma: no cover - makes this an async generator

    async def fake_gemini(prompt):
        gemini_called.append(prompt)
        yield "from-gemini"

    monkeypatch.setattr(llm_service, "_stream_with_groq", failing_groq)
    monkeypatch.setattr(llm_service, "_stream_with_gemini", fake_gemini)

    chunks = [c async for c in llm_service.stream_generate_answer("hello")]
    assert chunks == ["from-gemini"]
    assert gemini_called == ["hello"]


async def test_no_fallback_after_partial_stream(providers, monkeypatch):
    """
    Regression: replaying via a second provider after chunks were already
    sent would duplicate the answer. Once anything was yielded, the error
    must propagate instead.
    """
    gemini_called = []

    async def partial_groq(prompt):
        yield "partial-answer"
        raise HTTPException(status_code=502, detail="connection reset")

    async def fake_gemini(prompt):
        gemini_called.append(prompt)
        yield "from-gemini"

    monkeypatch.setattr(llm_service, "_stream_with_groq", partial_groq)
    monkeypatch.setattr(llm_service, "_stream_with_gemini", fake_gemini)

    received = []
    with pytest.raises(HTTPException):
        async for chunk in llm_service.stream_generate_answer("hello"):
            received.append(chunk)

    assert received == ["partial-answer"]
    assert gemini_called == []


def test_get_llm_model_follows_provider(providers, monkeypatch):
    assert llm_service.get_llm_provider() == "groq"
    assert llm_service.get_llm_model() == llm_service.GROQ_CHAT_MODEL
    monkeypatch.setattr(llm_service, "GROQ_API_KEY", None)
    assert llm_service.get_llm_provider() == "gemini"
    assert llm_service.get_llm_model() == llm_service.GEMINI_CHAT_MODEL


async def test_groq_stream_parses_sse_and_skips_malformed_frames(providers, monkeypatch):
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

    chunks = [c async for c in llm_service._stream_with_groq("prompt")]
    assert chunks == ["Hi", " there"]
    await test_client.aclose()


async def test_generate_answer_falls_back_to_gemini_when_groq_fails(providers, monkeypatch):
    async def failing_generate(prompt):
        raise HTTPException(status_code=502, detail="groq down")

    async def fake_gemini(prompt):
        return "gemini-answer"

    monkeypatch.setattr(llm_service, "_generate_with_groq", failing_generate)
    monkeypatch.setattr(llm_service, "_generate_with_gemini", fake_gemini)

    assert await llm_service.generate_answer("prompt") == "gemini-answer"


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
