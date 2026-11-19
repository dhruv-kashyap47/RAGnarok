"""
embedding_service.py — Groq batch embedding client (OpenAI-compatible).

Sends text chunks to Groq's POST /embeddings endpoint and returns
EMBEDDING_DIMENSIONS-dimensional float vectors ready for pgvector storage.
"""

from __future__ import annotations

import math

import httpx
from fastapi import HTTPException

from app.core.config import (
    EMBEDDING_DIMENSIONS,
    GROQ_API_BASE,
    GROQ_API_KEY,
    GROQ_EMBED_MODEL,
    HTTP_TIMEOUT_SECONDS,
)
from app.core.logger import logger

# Shared async client — API key is injected per-request so it is always fresh.
# Created lazily so it is bound to the running event loop and can be replaced
# after shutdown (e.g. across test lifespans).
_client: httpx.AsyncClient | None = None

# Groq embeddings endpoint batch limit (in texts per request).
_MAX_BATCH_SIZE = 100


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
    return _client


async def close_http_client() -> None:
    """Close the shared HTTP client on application shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def get_embedding_provider() -> str:
    return "groq"


def get_embedding_model() -> str:
    return GROQ_EMBED_MODEL


def get_embedding_dimensions() -> int:
    return EMBEDDING_DIMENSIONS


def _extract_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Groq request failed"
    error = payload.get("error") or {}
    return error.get("message") or response.text or "Groq request failed"


def _validate_embedding(values: list[float]) -> list[float]:
    if len(values) != EMBEDDING_DIMENSIONS:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Groq returned {len(values)} dimensions; "
                f"expected {EMBEDDING_DIMENSIONS}. "
                f"Ensure EMBEDDING_DIMENSIONS matches the model."
            ),
        )
    for value in values:
        # Reject booleans, strings, NaN/inf: anything unexpected would poison
        # the vector column or break the similarity-search literal.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(
                status_code=502,
                detail="Groq returned non-numeric embedding values.",
            )
        if not math.isfinite(value):
            raise HTTPException(
                status_code=502,
                detail="Groq returned non-finite embedding values.",
            )
    return [float(v) for v in values]


async def _batch_embed(texts: list[str]) -> list[list[float]]:
    """Call /embeddings for a single batch (≤ _MAX_BATCH_SIZE texts)."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    try:
        response = await _get_client().post(
            f"{GROQ_API_BASE}/embeddings",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_EMBED_MODEL, "input": texts},
        )
    except httpx.HTTPError as exc:
        logger.error("Groq embedding transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Groq embeddings request failed. Please try again later.",
        ) from exc

    if response.status_code >= 400:
        error_msg = _extract_error(response)
        logger.error(
            "Groq embedding API error status=%s body=%s",
            response.status_code,
            error_msg,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Groq embeddings failed: {error_msg}",
        )

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="Groq embeddings response malformed")
    if len(data) != len(texts):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Groq returned {len(data)} embeddings "
                f"for {len(texts)} inputs"
            ),
        )

    # OpenAI-style responses may be out of order: restore request order via
    # the per-item `index` before pairing vectors with their source texts.
    try:
        ordered = sorted(data, key=lambda item: item.get("index", 0))
    except (AttributeError, TypeError):
        raise HTTPException(status_code=502, detail="Groq embeddings response malformed")

    embeddings: list[list[float]] = []
    for item in ordered:
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            raise HTTPException(
                status_code=502, detail="Groq embeddings response malformed"
            )
        embeddings.append(_validate_embedding(item["embedding"]))
    return embeddings


async def get_embeddings(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed a list of texts, automatically batching to respect API limits.
    Returns one embedding vector per input text.

    ``task_type`` is accepted for API compatibility with existing callers;
    Groq's embeddings endpoint does not take a task type.
    """
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return []

    logger.info(
        "Requesting embeddings for %d chunks (model=%s)",
        len(cleaned),
        GROQ_EMBED_MODEL,
    )

    results: list[list[float]] = []
    for i in range(0, len(cleaned), _MAX_BATCH_SIZE):
        batch = cleaned[i : i + _MAX_BATCH_SIZE]
        batch_results = await _batch_embed(batch)
        results.extend(batch_results)
        logger.debug("Embedded batch %d-%d of %d", i + 1, i + len(batch), len(cleaned))

    return results


async def get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """Embed a single text string."""
    embeddings = await get_embeddings([text], task_type=task_type)
    if not embeddings:
        raise HTTPException(status_code=400, detail="Text for embedding cannot be empty.")
    return embeddings[0]
