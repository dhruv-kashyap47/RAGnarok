"""Unit tests for vector-store write batching (no database required)."""

import pytest

from app.services import vector_store


class _FakeResult:
    def first(self):
        return None

    def fetchall(self):
        return []


class _FakeDB:
    """Records Core insert parameter batches instead of touching a database."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    async def execute(self, statement, params=None):
        if params is not None:
            self.calls.append(params)
        return _FakeResult()


async def test_store_documents_bulk_batches_to_bound_statement_size():
    db = _FakeDB()
    # Small fake vectors: the batching logic does not validate dimensions.
    chunks = [(f"chunk {i}", [0.1, 0.2, 0.3]) for i in range(1200)]

    inserted = await vector_store.store_documents_bulk(db, "user-1", "doc-1", chunks)

    assert inserted == 1200
    assert [len(call) for call in db.calls] == [500, 500, 200]
    # Rows carry ownership + document linkage for tenant isolation.
    assert db.calls[0][0]["user_id"] == "user-1"
    assert db.calls[0][0]["document_id"] == "doc-1"
    assert db.calls[0][0]["content"] == "chunk 0"
    assert db.calls[0][0]["embedding"] == [0.1, 0.2, 0.3]


async def test_store_documents_bulk_with_no_chunks_is_noop():
    db = _FakeDB()
    assert await vector_store.store_documents_bulk(db, "user-1", "doc-1", []) == 0
    assert db.calls == []


async def test_delete_document_returns_false_without_header():
    """Ownership must be checked before any delete statement is issued."""

    executed = []

    class _DB:
        async def execute(self, statement, params=None):
            executed.append(statement)
            return _FakeResult()

    result = await vector_store.delete_document(_DB(), "user-a", "doc-x")
    assert result is False
    # Only the ownership lookup ran; no DELETE was issued.
    assert len(executed) == 1


async def test_delete_document_deletes_chunks_before_header():
    executed = []

    class _HeaderResult:
        def first(self):
            return ("doc-x",)

    class _DB:
        async def execute(self, statement, params=None):
            executed.append(str(statement))
            if "document_files" in str(statement) and "DELETE" in str(statement).upper():
                return _FakeResult()
            if "SELECT" in str(statement).upper():
                return _HeaderResult()
            return _FakeResult()

    result = await vector_store.delete_document(_DB(), "user-a", "doc-x")
    assert result is True
    deletes = [s for s in executed if "DELETE" in s.upper()]
    assert len(deletes) == 2
    # Chunks first (FK to document_files.id), then the header.
    assert "documents" in deletes[0]
    assert "document_files" in deletes[1]


@pytest.mark.parametrize("k", [1, 4, 10])
async def test_similarity_search_passes_limit_as_parameter(k):
    captured = {}

    class _DB:
        async def execute(self, statement, params=None):
            captured["sql"] = str(statement)
            captured["params"] = params
            return _FakeResult()

    await vector_store.similarity_search(_DB(), [0.1, 0.2], user_id="u1", k=k)
    assert captured["params"]["k"] == k
    # The embedding goes in as a bound parameter, never string-interpolated SQL.
    assert ":embedding" in captured["sql"]
    assert captured["params"]["embedding"].startswith("[")
