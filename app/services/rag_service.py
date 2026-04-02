import json
from fastapi import HTTPException

from app.core.config import MAX_CHAT_HISTORY_MESSAGES
from app.core.logger import logger
from app.schemas.chat import ChatResponse, ChatSource
from app.services.embedding_service import get_embedding
from app.services.llm_service import generate_answer, stream_generate_answer
from app.services.vector_store import similarity_search


def _format_history(history: list[dict]) -> str:
    trimmed = history[-MAX_CHAT_HISTORY_MESSAGES:]
    if not trimmed:
        return "No prior conversation."

    lines = []
    for item in trimmed:
        role = "User" if item.get("role") == "user" else "Assistant"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "No prior conversation."


def _build_prompt(question: str, history: list[dict], results: list[dict]) -> str:
    history_text = _format_history(history)

    if not results:
        return f"""You are a helpful assistant.
Answer clearly and directly.
Use the conversation history if it helps.
If you are unsure, say so briefly.

Conversation history:
{history_text}

Question:
{question}"""

    context = "\n\n".join(
        f"Source: {item['filename'] or 'Unknown file'}\n{item['content']}"
        for item in results
    )

    return f"""You are a helpful RAG assistant.
Answer the user's question using the retrieved context first.
If the retrieved context is incomplete, still give the best direct answer you can.
Be concise and useful.

Conversation history:
{history_text}

Retrieved context:
{context}

Question:
{question}"""


def _collect_sources(results: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for item in results:
        key = (item["document_id"], item["filename"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {"document_id": item["document_id"], "filename": item["filename"]}
        )
    return sources


def _normalize_history(history: list[dict] | None) -> list[dict]:
    normalized = []
    for item in history or []:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized[-MAX_CHAT_HISTORY_MESSAGES:]


async def ask_question_stream(
    db,
    question: str,
    user_id: str,
    history: list[dict] | None = None,
    k: int = 4,
):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = _normalize_history(history)
    results = []
    warning = None
    retrieval_status = "empty"
    logger.info("ask_question stream user_id=%s question=%r", user_id, question[:120])

    try:
        embedding = await get_embedding(question, task_type="RETRIEVAL_QUERY")
        results = await similarity_search(db, embedding, user_id=user_id, k=k)
        retrieval_status = "used" if results else "empty"
    except Exception as exc:
        logger.warning("Retrieval failed for user_id=%s: %s", user_id, exc)
        warning = (
            "Document retrieval was unavailable, so this answer may rely on model "
            "knowledge instead of your indexed files."
        )
        retrieval_status = "failed"

    # initial payload
    initial_metadata = {
        "type": "metadata",
        "sources": [{"document_id": item["document_id"], "filename": item["filename"]} for item in _collect_sources(results)],
        "used_context": bool(results),
        "retrieval_status": retrieval_status,
        "warning": warning,
    }
    yield json.dumps(initial_metadata) + "\n"

    try:
        async for chunk in stream_generate_answer(_build_prompt(question, history, results)):
            yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
    except Exception as exc:
        logger.error("Generation failed for user_id=%s: %s", user_id, exc)
        yield json.dumps({"type": "chunk", "content": "\n[Error: Stream generation failed]"}) + "\n"
