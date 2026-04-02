"""
vector_store.py — Database layer for document storage and pgvector similarity search.
"""

from __future__ import annotations

import time
from typing import TypedDict
from uuid import UUID

from sqlalchemy import select, text, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import logger
from app.models.documents import Document, DocumentFile


class DocumentFileSummary(TypedDict):
    id: str
    filename: str
    created_at: str | None


class SearchResult(TypedDict):
    content: str
    document_id: str | None
    filename: str | None


_SLOW_QUERY_MS = 300
_MAX_COSINE_DISTANCE = 0.45


async def create_document_record(
    db: AsyncSession,
    user_id: str | UUID,
    filename: str,
) -> DocumentFile:
    """Insert a DocumentFile header row and return it."""
    doc_file = DocumentFile(user_id=user_id, filename=filename)
    db.add(doc_file)
    await db.flush()
    return doc_file


async def store_document(
    db: AsyncSession,
    user_id: str | UUID,
    document_id: str | UUID,
    content: str,
    embedding: list[float],
) -> Document:
    """Insert a single Document chunk."""
    doc = Document(
        user_id=user_id,
        document_id=document_id,
        content=content,
        embedding=embedding,
    )
    db.add(doc)
    await db.flush()
    return doc


async def store_documents_bulk(
    db: AsyncSession,
    user_id: str | UUID,
    document_id: str | UUID,
    chunks: list[tuple[str, list[float]]],
) -> int:
    """Insert multiple (content, embedding) chunks in a single round-trip."""
    if not chunks:
        return 0

    rows = [
        Document(
            user_id=user_id,
            document_id=document_id,
            content=content,
            embedding=embedding,
        )
        for content, embedding in chunks
    ]
    db.add_all(rows)
    await db.flush()
    logger.debug("Bulk-inserted %d chunks for document_id=%s", len(rows), document_id)
    return len(rows)


async def list_documents(
    db: AsyncSession,
    user_id: str | UUID,
) -> list[DocumentFileSummary]:
    """Return all DocumentFile rows for user_id, newest first."""
    result = await db.execute(
        select(DocumentFile)
        .where(DocumentFile.user_id == user_id)
        .order_by(DocumentFile.created_at.desc())
    )
    return [
        DocumentFileSummary(
            id=str(doc.id),
            filename=doc.filename,
            created_at=doc.created_at.isoformat() if doc.created_at else None,
        )
        for doc in result.scalars().all()
    ]


async def delete_document(
    db: AsyncSession,
    user_id: str | UUID,
    document_id: str | UUID,
) -> bool:
    """Delete a DocumentFile and all its Document chunks for a given user."""
    # Delete chunks first
    await db.execute(
        delete(Document)
        .where(Document.document_id == document_id)
        .where(Document.user_id == user_id)
    )
    # Delete file record
    result = await db.execute(
        delete(DocumentFile)
        .where(DocumentFile.id == document_id)
        .where(DocumentFile.user_id == user_id)
    )
    return result.rowcount > 0


async def similarity_search(
    db: AsyncSession,
    query_embedding: list[float],
    user_id: str | UUID,
    k: int = 4,
    max_distance: float = _MAX_COSINE_DISTANCE,
) -> list[SearchResult]:
    """Return the k most relevant chunks using pgvector cosine distance."""
    t0 = time.monotonic()

    embedding_literal = f"[{','.join(map(str, query_embedding))}]"

    sql = text("""
        SELECT
            d.content,
            d.document_id,
            f.filename,
            (d.embedding <=> CAST(:embedding AS vector)) AS distance
        FROM documents d
        LEFT JOIN document_files f ON f.id = d.document_id
        WHERE d.user_id = :user_id
          AND (d.embedding <=> CAST(:embedding AS vector)) < :max_distance
        ORDER BY distance
        LIMIT :k
    """)

    result = await db.execute(
        sql,
        {
            "embedding": embedding_literal,
            "user_id": str(user_id),
            "max_distance": max_distance,
            "k": k,
        },
    )

    rows = result.fetchall()

    elapsed_ms = (time.monotonic() - t0) * 1000
    if elapsed_ms > _SLOW_QUERY_MS:
        logger.warning("similarity_search took %.0f ms (user_id=%s)", elapsed_ms, user_id)

    return [
        SearchResult(
            content=row[0],
            document_id=str(row[1]) if row[1] else None,
            filename=row[2],
        )
        for row in rows
    ]
