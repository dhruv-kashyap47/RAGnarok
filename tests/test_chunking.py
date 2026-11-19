"""Unit tests for the word-boundary chunker."""

from app.services.chunking import chunk_text


def test_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_short_text_is_single_chunk():
    text = "hello world"
    assert chunk_text(text, chunk_size=100, overlap=10) == ["hello world"]


def test_words_are_never_split():
    words = [f"word{i}" for i in range(2000)]
    chunks = chunk_text(" ".join(words), chunk_size=200, overlap=40)
    assert len(chunks) > 1
    original = set(words)
    # Every token in every chunk must be an intact original word.
    for chunk in chunks:
        assert set(chunk.split()) <= original


def test_chunks_respect_size_budget_approximately():
    words = [f"word{i:04d}" for i in range(3000)]
    chunks = chunk_text(" ".join(words), chunk_size=500, overlap=100)
    # Word-boundary chunking may slightly overshoot for long words, but the
    # budget must be respected within a small margin.
    assert all(len(c) <= 500 + 50 for c in chunks)


def test_overlap_keeps_context_between_chunks():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert len(chunks) >= 2
    # Overlap is char-based: the start of chunk N+1 must reappear inside
    # chunk N (words are never split, so whole-word overlap is preserved).
    first_words = set(chunks[0].split())
    head_next = chunks[1].split()
    assert head_next[0] in first_words
    assert all(word in first_words for word in head_next[:5])
