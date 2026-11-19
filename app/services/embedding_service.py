"""
embedding_service.py — Gemini batch embedding client.

Sends text chunks to Gemini's batchEmbedContents endpoint and returns
EMBEDDING_DIMENSIONS-dimensional float vectors ready for pgvector storage.

Groq is used for chat only: its `nomic-embed-text-v1_5` embedding model is
not enabled for regular API keys (POST /openai/v1/embeddings returns 404),
which is why embeddings live here instead.
"""

from __future__ import annotations

import asyncio
import math

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

# Shared async client — API key is injected per-request so it is always fresh.
# Created lazily so it is bound to the running event loop and can be replaced
# after shutdown (e.g. across test lifespans).
_client: httpx.AsyncClient | None = None

# Gemini embedding model hard limit per request (in texts, not tokens).
_MAX_BATCH_SIZE = 100

# Start-up self-check timeout: keep it short so a slow/unreachable provider
# cannot delay (or trip) the platform's port-bind deadline.
_PROBE_TIMEOUT_SECONDS = 10.0

# Result of the most recent self-check, surfaced by /health and /capabilities.
_health: dict[str, object] = {
    "ok": None,
    "provider": "gemini",
    "model": None,
    "dimensions": None,
    "error": None,
}


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
    return "gemini"


def get_embedding_model() -> str:
    return GEMINI_EMBED_MODEL


def get_embedding_dimensions() -> int:
    return EMBEDDING_DIMENSIONS


def get_embedding_health() -> dict[str, object]:
    """Snapshot of the last start-up probe (``ok`` is None until it runs)."""
    return {
        **_health,
        "provider": get_embedding_provider(),
        "model": GEMINI_EMBED_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
    }


def _extract_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "Gemini request failed").strip()
    error = payload.get("error") or {}
    return error.get("message") or response.text or "Gemini request failed"


def _normalize(values: list[float]) -> list[float]:
    """L2-normalize a vector (Gemini recommends this below 3072 dims)."""
    norm = math.sqrt(sum(v * v for v in values))
    if not norm or not math.isfinite(norm):
        return values
    return [v / norm for v in values]


def _validate_embedding(values: list[float]) -> list[float]:
    if len(values) < EMBEDDING_DIMENSIONS:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Gemini returned {len(values)} dimensions; "
                f"expected {EMBEDDING_DIMENSIONS}. "
                f"Ensure EMBEDDING_DIMENSIONS matches the model."
            ),
        )
    if len(values) > EMBEDDING_DIMENSIONS:
        # gemini-embedding-001 is Matryoshka-trained, so the leading segment
        # is a valid lower-dimensional embedding. Used only when the API
        # ignores/rejects outputDimensionality.
        logger.warning(
            "Gemini returned %d dimensions; truncating to %d (Matryoshka)",
            len(values),
            EMBEDDING_DIMENSIONS,
        )
        values = values[:EMBEDDING_DIMENSIONS]

    for value in values:
        # Reject booleans, strings, NaN/inf: anything unexpected would poison
        # the vector column or break the similarity-search literal.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(
                status_code=502,
                detail="Gemini returned non-numeric embedding values.",
            )
        if not math.isfinite(value):
            raise HTTPException(
                status_code=502,
                detail="Gemini returned non-finite embedding values.",
            )

    return _normalize([float(v) for v in values])


def _requests_payload(texts: list[str], task_type: str, with_dimensions: bool):
    requests = []
    for text in texts:
        item = {
            "model": f"models/{GEMINI_EMBED_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        if with_dimensions:
            item["outputDimensionality"] = EMBEDDING_DIMENSIONS
        requests.append(item)
    return {"requests": requests}


def _dimensions_rejected(response: httpx.Response) -> bool:
    """True when a 400 is specifically about outputDimensionality."""
    message = _extract_error(response).lower()
    return "dimension" in message and (
        "unknown" in message
        or "unsupported" in message
        or "not supported" in message
        or "invalid" in message
    )


async def _batch_embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Call batchEmbedContents for a single batch (≤ _MAX_BATCH_SIZE texts)."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Embeddings are not configured: GEMINI_API_KEY is missing. "
                "Set it in the environment to enable PDF uploads and chat."
            ),
        )

    url = f"{GEMINI_API_BASE}/{GEMINI_EMBED_MODEL}:batchEmbedContents"
    headers = {"x-goog-api-key": GEMINI_API_KEY}

    async def _post(with_dimensions: bool) -> httpx.Response:
        return await _get_client().post(
            url,
            headers=headers,
            json=_requests_payload(texts, task_type, with_dimensions),
        )

    try:
        response = await _post(with_dimensions=True)
        if response.status_code == 400 and _dimensions_rejected(response):
            # Older API surfaces reject outputDimensionality on the batch
            # endpoint; retry without it and truncate the Matryoshka prefix.
            logger.warning(
                "Gemini rejected outputDimensionality; retrying without it "
                "and truncating vectors to %d dims",
                EMBEDDING_DIMENSIONS,
            )
            response = await _post(with_dimensions=False)
    except httpx.HTTPError as exc:
        logger.error("Gemini embedding transport error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Embedding provider unreachable. Please try again shortly.",
        ) from exc

    if response.status_code >= 400:
        error_msg = _extract_error(response)
        logger.error(
            "Gemini embedding API error status=%s body=%s",
            response.status_code,
            error_msg,
        )
        if response.status_code in (401, 403):
            detail = (
                "Embedding provider rejected the API key. "
                "Check that GEMINI_API_KEY is valid and enabled for the "
                "generativelanguage.googleapis.com API."
            )
        elif response.status_code == 429:
            detail = (
                "Embedding rate limit/quota exhausted. "
                "Retry shortly or raise the Gemini quota for this key."
            )
        elif response.status_code == 404:
            detail = (
                f"Embedding model '{GEMINI_EMBED_MODEL}' is unavailable for "
                f"this key. Set GEMINI_EMBED_MODEL to a model the key can use."
            )
        else:
            detail = f"Embedding provider error ({response.status_code}): {error_msg}"
        raise HTTPException(status_code=503, detail=detail)

    payload = response.json()
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise HTTPException(
            status_code=502, detail="Gemini embeddings response malformed"
        )
    if len(embeddings) != len(texts):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Gemini returned {len(embeddings)} embeddings "
                f"for {len(texts)} inputs"
            ),
        )

    output: list[list[float]] = []
    for item in embeddings:
        if not isinstance(item, dict) or not isinstance(item.get("values"), list):
            raise HTTPException(
                status_code=502, detail="Gemini embeddings response malformed"
            )
        output.append(_validate_embedding(item["values"]))
    return output


async def get_embeddings(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed a list of texts, automatically batching to respect API limits.
    Returns one embedding vector per input text.
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


async def probe_embeddings() -> dict[str, object]:
    """
    Verify the embedding provider works, once, at start-up.

    Never raises: a broken provider must not stop the web process (and make
    the platform restart-loop). The result is surfaced by /health so a
    misconfiguration is obvious instead of only failing on the first upload.
    """
    global _health
    try:
        vector = await asyncio.wait_for(
            get_embedding("embedding health check"),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        _health = {
            "ok": True,
            "provider": get_embedding_provider(),
            "model": GEMINI_EMBED_MODEL,
            "dimensions": len(vector),
            "error": None,
        }
        logger.info(
            "Embeddings self-check passed (provider=%s model=%s dims=%d)",
            get_embedding_provider(),
            GEMINI_EMBED_MODEL,
            len(vector),
        )
    except TimeoutError:
        _health = {
            "ok": False,
            "provider": get_embedding_provider(),
            "model": GEMINI_EMBED_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "error": (
                "Embedding provider did not respond within "
                f"{_PROBE_TIMEOUT_SECONDS:.0f}s"
            ),
        }
        logger.error("Embeddings self-check FAILED: %s", _health["error"])
    except HTTPException as exc:
        _health = {
            "ok": False,
            "provider": get_embedding_provider(),
            "model": GEMINI_EMBED_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "error": str(exc.detail),
        }
        logger.error("Embeddings self-check FAILED: %s", exc.detail)
    except Exception as exc:  # noqa: BLE001 - never break start-up
        _health = {
            "ok": False,
            "provider": get_embedding_provider(),
            "model": GEMINI_EMBED_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "error": type(exc).__name__,
        }
        logger.exception("Embeddings self-check crashed: %s", exc)
    return get_embedding_health()
