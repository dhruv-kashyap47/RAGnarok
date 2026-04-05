"""
embedding_service.py — Gemini batch embedding client.

Sends text chunks to the Gemini batchEmbedContents API and returns
768-dimensional float vectors ready for pgvector storage.
"""

from __future__ import annotations

import httpx
from fastapi import HTTPException

from app.core.config import (
    EMBEDDING_DIMENSIONS,
    GEMINI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
    HTTP_TIMEOUT_SECONDS,
)
from app.core.logger import logger

# Shared async client — API key is injected per-request so it is always fresh
_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)

# Gemini embedding-001 hard limit per request (in texts, not tokens)
_MAX_BATCH_SIZE = 100


def get_embedding_provider() -> str:
    return "gemini"


def get_embedding_dimensions() -> int:
    return EMBEDDING_DIMENSIONS


def _extract_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Gemini request failed"
    error = payload.get("error") or {}
    return error.get("message") or response.text or "Gemini request failed"


def _validate_embedding(values: list[float]) -> list[float]:
    if len(values) != EMBEDDING_DIMENSIONS:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Gemini returned {len(values)} dimensions; "
                f"expected {EMBEDDING_DIMENSIONS}. "
                f"Ensure EMBEDDING_DIMENSIONS matches the model."
            ),
        )
    return values


async def _batch_embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Call batchEmbedContents for a single batch (≤ _MAX_BATCH_SIZE texts)."""
    api_key = GEMINI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    try:
        response = await _client.post(
            f"{GEMINI_API_BASE}/{GEMINI_EMBED_MODEL}:batchEmbedContents",
            headers={"x-goog-api-key": api_key},
            json={
                "requests": [
                    {
                        "model": f"models/{GEMINI_EMBED_MODEL}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": task_type,
                    }
                    for text in texts
                ]
            },
        )
    except httpx.HTTPError as exc:
        logger.error("Gemini embedding transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini embeddings request failed: {exc}",
        ) from exc

    if response.status_code >= 400:
        error_msg = _extract_error(response)
        logger.error(
            "Gemini embedding API error status=%s body=%s",
            response.status_code,
            error_msg,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Gemini embeddings failed: {error_msg}",
        )

    payload = response.json()
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise HTTPException(status_code=502, detail="Gemini embeddings response malformed")
    if len(embeddings) != len(texts):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Gemini returned {len(embeddings)} embeddings "
                f"for {len(texts)} inputs"
            ),
        )

    return [_validate_embedding(item.get("values", [])) for item in embeddings]


async def get_embeddings(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed a list of texts, automatically batching to respect API limits.
    Returns one embedding vector per text (filtered empties removed).
    """
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return []

    logger.info(
        "Requesting embeddings for %d chunks (task_type=%s, model=%s)",
        len(cleaned),
        task_type,
        GEMINI_EMBED_MODEL,
    )

    results: list[list[float]] = []
    for i in range(0, len(cleaned), _MAX_BATCH_SIZE):
        batch = cleaned[i : i + _MAX_BATCH_SIZE]
        batch_results = await _batch_embed(batch, task_type)
        results.extend(batch_results)
        logger.debug("Embedded batch %d-%d of %d", i + 1, i + len(batch), len(cleaned))

    return results


async def get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """Embed a single text string."""
    embeddings = await get_embeddings([text], task_type=task_type)
    if not embeddings:
        raise HTTPException(status_code=400, detail="Text for embedding cannot be empty.")
    return embeddings[0]
