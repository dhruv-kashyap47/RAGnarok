"""API tests: PDF upload pipeline, listing, deletion and tenant isolation."""

import uuid

from tests.conftest import make_pdf

SAMPLE_TEXT = (
    "Quarterly revenue grew by 18 percent driven by the Atlas segment. "
    "The board approved a new buyback program worth 400 million dollars. "
    "Headcount increased to 1200 employees across four regions."
)


def _upload(client, headers, text=SAMPLE_TEXT, filename="report.pdf", content_type="application/pdf"):
    return client.post(
        "/upload-pdf",
        headers=headers,
        files={"file": (filename, make_pdf(text), content_type)},
    )


def test_upload_requires_auth(client, mock_embeddings):
    response = _upload(client, {})
    assert response.status_code == 401


def test_upload_and_list(client, account, mock_embeddings, db_conn):
    response = _upload(client, account["headers"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["message"] == "PDF processed successfully"
    assert body["filename"] == "report.pdf"
    assert body["chunks"] >= 1

    documents = client.get("/documents", headers=account["headers"])
    assert documents.status_code == 200
    listed = documents.json()
    assert len(listed) == 1
    assert listed[0]["id"] == body["document_id"]
    assert listed[0]["filename"] == "report.pdf"
    assert listed[0]["created_at"]

    # Header row and chunk rows were committed together.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM documents WHERE document_id = %s",
            (body["document_id"],),
        )
        assert cur.fetchone()[0] == body["chunks"]
        cur.execute(
            "SELECT count(*) FROM document_files WHERE id = %s",
            (body["document_id"],),
        )
        assert cur.fetchone()[0] == 1


def test_upload_rejects_non_pdf_extension(client, account, mock_embeddings):
    response = _upload(client, account["headers"], filename="notes.txt")
    assert response.status_code == 400
    assert "PDF" in response.json()["message"]


def test_upload_rejects_wrong_content_type(client, account, mock_embeddings):
    response = _upload(client, account["headers"], filename="report.pdf", content_type="text/html")
    assert response.status_code == 400
    assert "content type" in response.json()["message"].lower()


def test_upload_rejects_empty_file(client, account, mock_embeddings):
    response = client.post(
        "/upload-pdf",
        headers=account["headers"],
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["message"].lower()


def test_upload_enforces_pdf_size_limit(client, account, mock_embeddings, monkeypatch):
    """Regression: the file used to be read fully into memory before the size
    check; now it streams to disk and aborts as soon as the cap is exceeded."""
    import app.api.user as api_user

    monkeypatch.setattr(api_user, "_MAX_PDF_BYTES", 1024)
    big_pdf = make_pdf(SAMPLE_TEXT * 50)  # comfortably > 1 KB
    assert len(big_pdf) > 1024

    response = client.post(
        "/upload-pdf",
        headers=account["headers"],
        files={"file": ("big.pdf", big_pdf, "application/pdf")},
    )
    assert response.status_code == 413
    assert "size limit" in response.json()["message"]

    # Nothing was persisted for the rejected upload.
    assert client.get("/documents", headers=account["headers"]).json() == []


def test_upload_failure_leaves_no_partial_document(
    client, account, monkeypatch
):
    """A failure during embedding must roll back the header row too."""

    async def broken_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
        raise RuntimeError("embedding backend exploded")

    monkeypatch.setattr("app.api.user.get_embeddings", broken_embeddings)

    response = _upload(client, account["headers"])
    assert response.status_code == 500
    assert response.json()["message"] == "An unexpected error occurred while processing the PDF."
    assert client.get("/documents", headers=account["headers"]).json() == []


def test_upload_embedding_count_mismatch_is_502(client, account, monkeypatch):
    async def short_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
        return []  # zero vectors for any number of chunks: always a mismatch

    monkeypatch.setattr("app.api.user.get_embeddings", short_embeddings)

    response = _upload(client, account["headers"])
    assert response.status_code == 502
    assert client.get("/documents", headers=account["headers"]).json() == []


def test_delete_own_document(client, account, mock_embeddings, db_conn):
    doc_id = _upload(client, account["headers"]).json()["document_id"]

    response = client.delete(f"/documents/{doc_id}", headers=account["headers"])
    assert response.status_code == 200
    assert response.json()["message"] == "Document deleted successfully"

    assert client.get("/documents", headers=account["headers"]).json() == []
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents WHERE document_id = %s", (doc_id,))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM document_files WHERE id = %s", (doc_id,))
        assert cur.fetchone()[0] == 0


def test_delete_with_malformed_id_returns_404_not_500(client, account):
    """Regression: non-UUID ids used to reach the DB driver and explode
    as an unhandled 500."""
    response = client.delete("/documents/not-a-uuid", headers=account["headers"])
    assert response.status_code == 404
    assert response.json()["message"] == "Document not found"


def test_delete_unknown_id_returns_404(client, account):
    unknown = str(uuid.uuid4())
    response = client.delete(f"/documents/{unknown}", headers=account["headers"])
    assert response.status_code == 404


def test_delete_requires_auth(client, account, mock_embeddings):
    doc_id = _upload(client, account["headers"]).json()["document_id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 401


def test_cross_tenant_delete_is_blocked(client, account, mock_embeddings, db_conn):
    """IDOR: another user must not be able to delete someone else's document."""
    owner = account
    intruder_email = f"intruder-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/users", json={"email": intruder_email, "password": "secret123"})
    intruder_token = client.post(
        "/login", json={"email": intruder_email, "password": "secret123"}
    ).json()["access_token"]
    intruder = {"Authorization": f"Bearer {intruder_token}"}

    doc_id = _upload(client, owner["headers"]).json()["document_id"]

    response = client.delete(f"/documents/{doc_id}", headers=intruder)
    assert response.status_code == 404

    # Owner's data is untouched: header and chunks both survive.
    assert len(client.get("/documents", headers=owner["headers"]).json()) == 1
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents WHERE document_id = %s", (doc_id,))
        assert cur.fetchone()[0] >= 1


def test_cross_tenant_listing_is_isolated(client, account, mock_embeddings):
    intruder_email = f"lurker-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/users", json={"email": intruder_email, "password": "secret123"})
    intruder_token = client.post(
        "/login", json={"email": intruder_email, "password": "secret123"}
    ).json()["access_token"]
    intruder = {"Authorization": f"Bearer {intruder_token}"}

    _upload(client, account["headers"])

    assert len(client.get("/documents", headers=account["headers"]).json()) == 1
    assert client.get("/documents", headers=intruder).json() == []


def test_hostile_filename_is_stored_not_used_as_path(client, account, mock_embeddings):
    """A traversal-style filename must never be used to build a file path."""
    evil_name = "../../evil.pdf"
    response = _upload(client, account["headers"], filename=evil_name)
    assert response.status_code == 200, response.text

    listed = client.get("/documents", headers=account["headers"]).json()
    assert listed[0]["filename"] == evil_name  # stored verbatim as data

    # The upload was processed in a temp directory, not at a relative path.
    import os

    assert not os.path.exists(os.path.join(os.getcwd(), "evil.pdf"))
