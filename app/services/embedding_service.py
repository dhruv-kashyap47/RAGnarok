import httpx
from fastapi import HTTPException

from app.core.config import (
    EMBEDDING_DIMENSIONS,
    GEMINI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
    HTTP_TIMEOUT_SECONDS,
)

_client = httpx.AsyncClient(
    timeout=HTTP_TIMEOUT_SECONDS,
    headers={"x-goog-api-key": GEMINI_API_KEY or ""},
)


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
            detail=f"Gemini returned {len(values)} dims; expected {EMBEDDING_DIMENSIONS}",
        )
    return values


async def get_embeddings(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return []

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    try:
        response = await _client.post(
            f"{GEMINI_API_BASE}/{GEMINI_EMBED_MODEL}:batchEmbedContents",
            json={
                "requests": [
                    {
                        "model": f"models/{GEMINI_EMBED_MODEL}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": task_type,
                        "outputDimensionality": EMBEDDING_DIMENSIONS,
                    }
                    for text in cleaned
                ]
            },
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini embeddings request failed: {exc}",
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini embeddings failed: {_extract_error(response)}",
        )

    payload = response.json()
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise HTTPException(status_code=502, detail="Gemini embeddings response malformed")
    if len(embeddings) != len(cleaned):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Gemini returned {len(embeddings)} embeddings for "
                f"{len(cleaned)} inputs"
            ),
        )

    return [_validate_embedding(item.get("values", [])) for item in embeddings]


async def get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    embeddings = await get_embeddings([text], task_type=task_type)
    if not embeddings:
        raise HTTPException(status_code=400, detail="Text for embedding cannot be empty.")
    return embeddings[0]
