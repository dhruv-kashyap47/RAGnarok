"""
chunking.py — Word-boundary-aware text chunker for RAG ingestion.

Uses word-level splitting so chunks are never cut mid-word.
chunk_size and overlap are measured in *characters* (post-join).
"""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* chars.

    Words are never split mid-token. Leading/trailing whitespace is stripped.
    Empty or whitespace-only chunks are discarded.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word) + (1 if current else 0)  # +1 for the space separator
        if current_len + word_len > chunk_size and current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            # Keep overlap: drop words from the front until we're within budget
            while current and current_len > overlap:
                removed = current.pop(0)
                current_len -= len(removed) + 1
            current_len = max(current_len, 0)

        current.append(word)
        current_len += word_len

    # Flush the last chunk
    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks
