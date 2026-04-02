def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Break text into smaller overlapping chunks for better LLM retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
