"""Unit tests for the Gemini embedding client: batching and strict validation."""

import httpx
import pytest
from fastapi import HTTPException

from app.core.config import EMBEDDING_DIMENSIONS
from app.services import embedding_service


def _client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture()
def api_key(monkeypatch):
    monkeypatch.setattr(embedding_service, "GEMINI_API_KEY", "test-key")


async def test_missing_api_key_raises_500(monkeypatch):
    monkeypatch.setattr(embedding_service, "GEMINI_API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 500


async def test_empty_input_short_circuits(api_key):
    assert await embedding_service.get_embeddings([]) == []
    assert await embedding_service.get_embeddings(["   ", ""]) == []


async def test_batch_embed_success(api_key, monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        body = request.content.decode()
        captured["body"] = body
        count = body.count('"text"')
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.5] * EMBEDDING_DIMENSIONS} for _ in range(count)]},
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    out = await embedding_service.get_embeddings(["alpha", "beta"])
    assert len(out) == 2
    assert all(len(vec) == EMBEDDING_DIMENSIONS for vec in out)
    assert "batchEmbedContents" in captured["url"]
    await test_client.aclose()


async def test_large_input_is_split_into_batches_of_100(api_key, monkeypatch):
    batches = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        count = body.count('"text"')
        batches.append(count)
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1] * EMBEDDING_DIMENSIONS} for _ in range(count)]},
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    texts = [f"chunk {i}" for i in range(250)]
    out = await embedding_service.get_embeddings(texts)
    assert batches == [100, 100, 50]
    assert len(out) == 250
    await test_client.aclose()


async def test_dimension_mismatch_is_502(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [{"values": [0.1] * 8}]})

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 502
    assert "dimensions" in exc.value.detail
    await test_client.aclose()


async def test_non_numeric_values_are_rejected(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [{"values": ["0.1"] * EMBEDDING_DIMENSIONS}]},
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 502
    assert "non-numeric" in exc.value.detail
    await test_client.aclose()


async def test_non_finite_values_are_rejected(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        # NaN cannot be serialized by strict JSON encoders; craft the body
        # manually to simulate a provider returning a non-finite value.
        values = ",".join(["NaN"] + ["0.1"] * (EMBEDDING_DIMENSIONS - 1))
        body = '{"embeddings": [{"values": [' + values + "]}]}"
        return httpx.Response(
            200, text=body, headers={"content-type": "application/json"}
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 502
    assert "non-finite" in exc.value.detail
    await test_client.aclose()


async def test_api_error_status_is_502_with_provider_message(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "quota exceeded"}},
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 502
    assert "quota exceeded" in exc.value.detail
    await test_client.aclose()


async def test_mismatched_embedding_count_is_502(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [{"values": [0.1] * EMBEDDING_DIMENSIONS}]})

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["one", "two"])
    assert exc.value.status_code == 502
    assert "for 2 inputs" in exc.value.detail
    await test_client.aclose()


async def test_transport_error_is_502_and_does_not_leak_internals(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to https://internal-host")

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 502
    assert "internal-host" not in exc.value.detail
    await test_client.aclose()
