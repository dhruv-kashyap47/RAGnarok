import json
import httpx
from fastapi import HTTPException

from app.core.config import (
    CHAT_TIMEOUT_SECONDS,
    GEMINI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GROQ_API_BASE,
    GROQ_API_KEY,
    GROQ_CHAT_MODEL,
)
from app.core.logger import logger

_client = httpx.AsyncClient(timeout=CHAT_TIMEOUT_SECONDS)


def get_llm_provider() -> str:
    if GROQ_API_KEY:
        return "groq"
    if GEMINI_API_KEY:
        return "gemini"
    return "unconfigured"


def get_llm_model() -> str:
    if get_llm_provider() == "groq":
        return GROQ_CHAT_MODEL
    return GEMINI_CHAT_MODEL


def _extract_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "LLM request failed"
    error = payload.get("error") or {}
    return error.get("message") or response.text or "LLM request failed"


def _require_configured_provider() -> str:
    provider = get_llm_provider()
    if provider == "unconfigured":
        raise HTTPException(
            status_code=500,
            detail="No LLM API key is configured. Set GROQ_API_KEY or GEMINI_API_KEY.",
        )
    return provider


def _has_gemini_fallback() -> bool:
    return bool(GEMINI_API_KEY)


def _gemini_text_from_payload(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "".join(texts).strip()


async def _generate_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    logger.info("Sending prompt to Groq model=%s", GROQ_CHAT_MODEL)
    try:
        response = await _client.post(
            f"{GROQ_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
            },
        )
    except httpx.HTTPError as exc:
        logger.error("Groq transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Groq generation request failed: {exc}",
        ) from exc

    if response.status_code >= 400:
        logger.error("Groq failed status=%s", response.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"Groq generation failed: {_extract_error(response)}",
        )

    data = response.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    logger.info("Groq generation completed")
    return text


async def _generate_with_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    logger.info("Sending prompt to Gemini model=%s", GEMINI_CHAT_MODEL)
    try:
        response = await _client.post(
            f"{GEMINI_API_BASE}/{GEMINI_CHAT_MODEL}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024,
                },
            },
        )
    except httpx.HTTPError as exc:
        logger.error("Gemini transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini generation request failed: {exc}",
        ) from exc

    if response.status_code >= 400:
        logger.error("Gemini failed status=%s", response.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini generation failed: {_extract_error(response)}",
        )

    text = _gemini_text_from_payload(response.json())
    logger.info("Gemini generation completed")
    return text


async def generate_answer(prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    provider = _require_configured_provider()
    if provider == "groq":
        try:
            return await _generate_with_groq(prompt)
        except HTTPException as exc:
            if _has_gemini_fallback():
                logger.warning("Groq generation failed; falling back to Gemini: %s", exc.detail)
                return await _generate_with_gemini(prompt)
            raise
    return await _generate_with_gemini(prompt)


async def _stream_with_groq(prompt: str):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    logger.info("Streaming prompt to Groq model=%s", GROQ_CHAT_MODEL)
    try:
        async with _client.stream(
            "POST",
            f"{GROQ_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
                "stream": True,
            },
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                logger.error("Groq failed status=%s body=%s", response.status_code, body)
                raise HTTPException(
                    status_code=502,
                    detail=f"Groq generation failed: {_extract_error(response)}",
                )

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        text_chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if text_chunk:
                            yield text_chunk
                    except Exception:
                        pass
    except httpx.HTTPError as exc:
        logger.error("Groq stream transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Groq generation request failed: {exc}",
        ) from exc


async def _stream_with_gemini(prompt: str):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    logger.info("Streaming prompt to Gemini model=%s", GEMINI_CHAT_MODEL)
    try:
        async with _client.stream(
            "POST",
            f"{GEMINI_API_BASE}/{GEMINI_CHAT_MODEL}:streamGenerateContent?alt=sse",
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024,
                },
            },
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                logger.error("Gemini failed status=%s body=%s", response.status_code, body)
                raise HTTPException(
                    status_code=502,
                    detail=f"Gemini generation failed: {_extract_error(response)}",
                )

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if not data_str.strip():
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                text_chunk = _gemini_text_from_payload(data)
                if text_chunk:
                    yield text_chunk
    except httpx.HTTPError as exc:
        logger.error("Gemini stream transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini generation request failed: {exc}",
        ) from exc


async def stream_generate_answer(prompt: str):
    """Yield chunks of text generated by the LLM."""
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    provider = _require_configured_provider()
    if provider == "groq":
        try:
            async for chunk in _stream_with_groq(prompt):
                yield chunk
            return
        except HTTPException as exc:
            if _has_gemini_fallback():
                logger.warning("Groq stream failed; falling back to Gemini: %s", exc.detail)
                async for chunk in _stream_with_gemini(prompt):
                    yield chunk
                return
            raise

    async for chunk in _stream_with_gemini(prompt):
        yield chunk
