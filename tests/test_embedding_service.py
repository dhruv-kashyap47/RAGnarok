"""Unit tests for the Gemini embedding client: batching and strict validation."""

import json
import math

import httpx
import pytest
from fastapi import HTTPException

from app.core.config import EMBEDDING_DIMENSIONS, GEMINI_EMBED_MODEL
from app.services import embedding_service


def _client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _unit_vector(size: int, head: float = 0.0) -> list[float]:
    """A unit-length vector so normalization leaves it unchanged."""
    vec = [0.0] * size
    vec[0] = 1.0
    if head:
        vec[1] = head
        norm = math.sqrt(1.0 + head * head)
        return [value / norm for value in vec]
    return vec


def _ok_payload(vectors: list[list[float]]) -> dict:
    """Gemini batchEmbedContents-style response body."""
    return {"embeddings": [{"values": vec} for vec in vectors]}


@pytest.fixture()
def api_key(monkeypatch):
    monkeypatch.setattr(embedding_service, "GEMINI_API_KEY", "test-gemini-key")


async def test_missing_api_key_raises_503(monkeypatch):
    monkeypatch.setattr(embedding_service, "GEMINI_API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 503
    assert "GEMINI_API_KEY" in exc.value.detail


async def test_empty_input_short_circuits(api_key):
    assert await embedding_service.get_embeddings([]) == []
    assert await embedding_service.get_embeddings(["   ", ""]) == []


async def test_batch_embed_success(api_key, monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(
            200,
            json=_ok_payload(
                [_unit_vector(EMBEDDING_DIMENSIONS) for _ in body["requests"]]
            ),
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    out = await embedding_service.get_embeddings(["alpha", "beta"])
    assert len(out) == 2
    assert all(len(vec) == EMBEDDING_DIMENSIONS for vec in out)
    assert captured["url"].endswith(
        f"/{GEMINI_EMBED_MODEL}:batchEmbedContents"
    )
    requests = captured["body"]["requests"]
    assert [item["content"]["parts"][0]["text"] for item in requests] == [
        "alpha",
        "beta",
    ]
    assert all(
        item["model"] == f"models/{GEMINI_EMBED_MODEL}" for item in requests
    )
    assert all(
        item["outputDimensionality"] == EMBEDDING_DIMENSIONS for item in requests
    )
    assert all(item["taskType"] == "RETRIEVAL_DOCUMENT" for item in requests)
    await test_client.aclose()


async def test_vectors_are_l2_normalized(api_key, monkeypatch):
    raw = [0.5] * EMBEDDING_DIMENSIONS

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_payload([raw]))

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    [vec] = await embedding_service.get_embeddings(["hello"])
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)
    await test_client.aclose()


async def test_large_input_is_split_into_batches_of_100(api_key, monkeypatch):
    batches = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        batches.append(len(body["requests"]))
        return httpx.Response(
            200,
            json=_ok_payload(
                [_unit_vector(EMBEDDING_DIMENSIONS) for _ in body["requests"]]
            ),
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    texts = [f"chunk {i}" for i in range(250)]
    out = await embedding_service.get_embeddings(texts)
    assert batches == [100, 100, 50]
    assert len(out) == 250
    await test_client.aclose()


async def test_rejected_output_dimensionality_retries_and_truncates(
    api_key, monkeypatch
):
    """Older API surfaces reject outputDimensionality; fall back to truncation."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "outputDimensionality" in body["requests"][0]:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            'Invalid JSON payload received. Unknown name '
                            '"outputDimensionality" at requests[0].'
                        )
                    }
                },
            )
        return httpx.Response(
            200, json=_ok_payload([_unit_vector(3072)])
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    [vec] = await embedding_service.get_embeddings(["hello"])
    assert len(calls) == 2
    assert "outputDimensionality" not in calls[1]["requests"][0]
    assert len(vec) == EMBEDDING_DIMENSIONS
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)
    await test_client.aclose()


async def test_dimension_shortfall_is_502(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_payload([[0.1] * 8]))

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
            json=_ok_payload([["0.1"] * EMBEDDING_DIMENSIONS]),
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


async def test_invalid_api_key_is_503_with_actionable_message(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"message": "API key not valid. Please pass a valid API key."}},
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 503
    assert "GEMINI_API_KEY" in exc.value.detail
    await test_client.aclose()


async def test_quota_error_is_503(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "quota exceeded"}},
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 503
    assert "quota" in exc.value.detail.lower()
    await test_client.aclose()


async def test_mismatched_embedding_count_is_502(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_payload([_unit_vector(EMBEDDING_DIMENSIONS)]),
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["one", "two"])
    assert exc.value.status_code == 502
    assert "for 2 inputs" in exc.value.detail
    await test_client.aclose()


async def test_malformed_response_is_502(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 502
    assert "malformed" in exc.value.detail
    await test_client.aclose()


async def test_transport_error_is_503_and_does_not_leak_internals(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to https://internal-host")

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    with pytest.raises(HTTPException) as exc:
        await embedding_service.get_embeddings(["hello"])
    assert exc.value.status_code == 503
    assert "internal-host" not in exc.value.detail
    await test_client.aclose()


async def test_probe_records_failure_without_raising(monkeypatch):
    monkeypatch.setattr(embedding_service, "GEMINI_API_KEY", None)

    health = await embedding_service.probe_embeddings()
    assert health["ok"] is False
    assert "GEMINI_API_KEY" in health["error"]
    assert embedding_service.get_embedding_health()["ok"] is False


async def test_probe_records_success(api_key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_ok_payload([_unit_vector(EMBEDDING_DIMENSIONS)])
        )

    test_client = _client_with(handler)
    monkeypatch.setattr(embedding_service, "_get_client", lambda: test_client)

    health = await embedding_service.probe_embeddings()
    assert health["ok"] is True
    assert health["dimensions"] == EMBEDDING_DIMENSIONS
    assert health["provider"] == "gemini"
    await test_client.aclose()
