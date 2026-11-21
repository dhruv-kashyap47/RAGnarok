import json

import httpx
from fastapi import HTTPException

from app.core.config import (
    CHAT_TIMEOUT_SECONDS,
    GROQ_API_BASE,
    GROQ_API_KEY,
    GROQ_CHAT_MODEL,
    GROQ_FALLBACK_MODELS,
)
from app.core.logger import logger

# Created lazily so it binds to the running event loop and can be cycled on
# shutdown (e.g. across test lifespans).
_client: httpx.AsyncClient | None = None

# The last model that produced output. Sticky so that once we fall back we keep
# using the working model instead of probing a retired one on every request.
_active_model: str | None = None

# Chat models reported by GET /models. Cached lazily as a last-resort fallback
# for the day Groq retires every configured model (it retires them regularly).
_discovered_models: list[str] | None = None

# Models that cannot serve /chat/completions: speech, audio transcription,
# safety classifiers and embedding models live behind their own endpoints.
_NON_CHAT_MODEL_MARKERS = (
    "whisper",
    "tts",
    "orpheus",
    "prompt-guard",
    "safeguard",
    "moderation",
    "embed",
)

# Error text that means "this model name no longer exists" rather than "this
# request was bad". Only then is it worth asking Groq which models do exist.
_MODEL_MISSING_MARKERS = (
    "model_not_found",
    "does not exist",
    "decommissioned",
    "not supported",
    "not available",
    "no longer supported",
)

_MAX_DISCOVERED_MODELS = 4


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=CHAT_TIMEOUT_SECONDS)
    return _client


async def close_http_client() -> None:
    """Close the shared HTTP client on application shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def get_llm_provider() -> str:
    """Groq is the only supported provider."""
    return "groq" if GROQ_API_KEY else "unconfigured"


def get_llm_model() -> str:
    """The model serving requests right now (the fallback, if one kicked in)."""
    return _active_model or GROQ_CHAT_MODEL


def _configured_models() -> list[str]:
    """Primary model first, then fallbacks; the known-good model jumps the queue."""
    ordered = [_active_model, GROQ_CHAT_MODEL, *GROQ_FALLBACK_MODELS]
    models: list[str] = []
    for model in ordered:
        if model and model not in models:
            models.append(model)
    return models


async def _discover_chat_models() -> list[str]:
    """Ask Groq which chat models this key can use (fallback of last resort)."""
    global _discovered_models
    if _discovered_models is not None:
        return _discovered_models

    try:
        response = await _get_client().get(
            f"{GROQ_API_BASE}/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        )
        if response.status_code >= 400:
            logger.warning(
                "Groq model discovery failed status=%s body=%s",
                response.status_code,
                response.text[:300],
            )
            found: list[str] = []
        else:
            payload = response.json()
            candidates = [
                str(item.get("id", ""))
                for item in payload.get("data", [])
                if isinstance(item, dict)
            ]
            found = [
                model
                for model in candidates
                if model
                and not any(
                    marker in model.lower() for marker in _NON_CHAT_MODEL_MARKERS
                )
            ][:_MAX_DISCOVERED_MODELS]
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Groq model discovery failed: %s", exc)
        found = []

    _discovered_models = found
    if found:
        logger.info("Discovered Groq chat models: %s", ", ".join(found))
    return found


def _mark_model_active(model: str) -> None:
    global _active_model
    if _active_model == model:
        return
    if _active_model:
        logger.warning("Groq chat model switched from %s to %s", _active_model, model)
    _active_model = model


def _payload(model: str, prompt: str, stream: bool = False) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    if "gpt-oss" in model.lower():
        # gpt-oss spends max_tokens on hidden reasoning; low effort keeps
        # answers fast and leaves room for the actual response.
        payload["reasoning_effort"] = "low"
    if stream:
        payload["stream"] = True
    return payload


def _looks_like_missing_model(status: int, detail: str) -> bool:
    if status == 404:
        return True
    lowered = (detail or "").lower()
    return any(marker in lowered for marker in _MODEL_MISSING_MARKERS)


class _ModelUnavailable(Exception):
    """A single model refused the request; another candidate may still work."""

    def __init__(self, model: str, status: int, detail: str):
        super().__init__(detail)
        self.model = model
        self.status = status
        self.detail = detail


def _extract_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "LLM request failed"
    error = payload.get("error") or {}
    return error.get("message") or response.text or "LLM request failed"


def _require_api_key() -> None:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")


def _final_error(errors: list[_ModelUnavailable]) -> HTTPException:
    if not errors:
        detail = "no chat model is available for this API key"
    elif len(errors) == 1:
        detail = errors[-1].detail
    else:
        detail = f"{errors[-1].detail} (tried {len(errors)} models)"
    return HTTPException(
        status_code=502,
        detail=f"Groq generation failed: {detail}",
    )


async def _generate_with_model(model: str, prompt: str) -> str:
    try:
        response = await _get_client().post(
            f"{GROQ_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=_payload(model, prompt),
        )
    except httpx.HTTPError as exc:
        logger.error("Groq transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Groq generation request failed. Please try again later.",
        ) from exc

    if response.status_code >= 400:
        raise _ModelUnavailable(model, response.status_code, _extract_error(response))

    try:
        data = response.json()
    except ValueError as exc:
        raise _ModelUnavailable(model, 502, "Groq returned unreadable JSON") from exc

    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


async def generate_answer(prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    _require_api_key()

    errors: list[_ModelUnavailable] = []
    missing_model = False
    candidates = _configured_models()
    discovered = False
    index = 0

    while True:
        if index >= len(candidates):
            # Every configured model refused. If the cause was a retired model
            # name, ask the provider what it still serves and try that.
            if discovered or not missing_model:
                break
            discovered = True
            known = set(candidates)
            candidates = [
                model for model in await _discover_chat_models() if model not in known
            ]
            index = 0
            if not candidates:
                break

        model = candidates[index]
        index += 1
        try:
            text = await _generate_with_model(model, prompt)
        except _ModelUnavailable as exc:
            logger.warning(
                "Groq model=%s unavailable status=%s: %s",
                model,
                exc.status,
                exc.detail,
            )
            errors.append(exc)
            missing_model = missing_model or _looks_like_missing_model(
                exc.status, exc.detail
            )
            continue

        _mark_model_active(model)
        logger.info("Groq generation completed model=%s", model)
        return text

    raise _final_error(errors)


async def _stream_with_model(model: str, prompt: str):
    """Yield text chunks from one model, failing before the first chunk only."""
    try:
        async with _get_client().stream(
            "POST",
            f"{GROQ_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=_payload(model, prompt, stream=True),
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                detail = _extract_error(response)
                logger.error(
                    "Groq model=%s failed status=%s body=%s",
                    model,
                    response.status_code,
                    body,
                )
                raise _ModelUnavailable(model, response.status_code, detail)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        text_chunk = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if text_chunk:
                            yield text_chunk
                    except (
                        json.JSONDecodeError,
                        IndexError,
                        TypeError,
                        AttributeError,
                        KeyError,
                    ):
                        # Malformed SSE frame (keepalive, truncated JSON):
                        # skip it and keep streaming.
                        logger.debug("Skipping malformed Groq stream frame")
    except httpx.HTTPError as exc:
        logger.error("Groq stream transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Groq generation request failed. Please try again later.",
        ) from exc


async def stream_generate_answer(prompt: str):
    """Yield chunks of text generated by the LLM (Groq SSE stream).

    If a model rejects the request before any text is emitted (retired model,
    rate limit, provider error), the next candidate is tried transparently.
    Once chunks have been sent there is no safe restart, so the error is raised.
    """
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    _require_api_key()

    logger.info("Streaming prompt to Groq models=%s", ", ".join(_configured_models()))

    errors: list[_ModelUnavailable] = []
    missing_model = False
    candidates = _configured_models()
    discovered = False
    index = 0

    while True:
        if index >= len(candidates):
            if discovered or not missing_model:
                break
            discovered = True
            known = set(candidates)
            candidates = [
                model for model in await _discover_chat_models() if model not in known
            ]
            index = 0
            if not candidates:
                break

        model = candidates[index]
        index += 1
        emitted = False
        try:
            async for chunk in _stream_with_model(model, prompt):
                if not emitted:
                    emitted = True
                    _mark_model_active(model)
                    logger.info("Groq stream started model=%s", model)
                yield chunk
        except _ModelUnavailable as exc:
            if emitted:
                # The client already rendered partial text; restarting would
                # duplicate or contradict it, so surface the failure instead.
                logger.error(
                    "Groq stream failed mid-answer model=%s status=%s: %s",
                    model,
                    exc.status,
                    exc.detail,
                )
                raise _final_error([exc]) from exc
            logger.warning(
                "Groq model=%s unavailable status=%s: %s",
                model,
                exc.status,
                exc.detail,
            )
            errors.append(exc)
            missing_model = missing_model or _looks_like_missing_model(
                exc.status, exc.detail
            )
            continue

        _mark_model_active(model)
        return

    raise _final_error(errors)
