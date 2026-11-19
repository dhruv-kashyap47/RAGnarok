"""
Test bootstrap.

IMPORTANT: environment variables are configured here before any `app.*`
module is imported, because `app.core.config` reads them at import time.
"""

import os
import shutil
import subprocess
import sys
import time
import uuid

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DB_HOST = "127.0.0.1"
TEST_DB_PORT = 5433
TEST_DB_NAME = "ragproject_test"
CONTAINER_NAME = "ragproject-test-postgres"

TEST_SYNC_DB_URL = (
    f"postgresql://ragproject:ragproject@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
)

# Must match Vector(3072) from migration f3e8f8a12345 / the documents model.
TEST_EMBED_DIM = 3072

# --- Environment (before any app import) ----------------------------------
os.environ["SECRET_KEY"] = "test-secret-key-" + "0123456789abcdef" * 4
os.environ["DATABASE_URL"] = (
    f"postgresql+asyncpg://ragproject:ragproject@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
)
os.environ["SQL_ECHO"] = "false"
# Never talk to real provider APIs from the test suite.
os.environ["GEMINI_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""
# Point at a closed port: caching stays off and reconnects are backed off.
os.environ["REDIS_URL"] = f"redis://{TEST_DB_HOST}:1"
# Small body cap keeps the 413 test fast; test uploads stay well under it.
os.environ["MAX_REQUEST_BODY_BYTES"] = str(1024 * 1024)
# Generous defaults so functional tests never trip limits; the dedicated
# rate-limit tests monkeypatch these down.
os.environ["AUTH_RATE_LIMIT_PER_MINUTE"] = "1000"
os.environ["REGISTER_RATE_LIMIT_PER_MINUTE"] = "1000"
os.environ["ASK_RATE_LIMIT_PER_MINUTE"] = "1000"
os.environ["UPLOAD_RATE_LIMIT_PER_MINUTE"] = "1000"
os.environ["CORS_ORIGINS"] = "*"
os.environ["ALLOW_CREDENTIALS"] = "false"


# --- Test database (Docker) -----------------------------------------------

def _docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _connect_admin(retries: int = 90):
    import psycopg2

    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            conn = psycopg2.connect(
                host=TEST_DB_HOST,
                port=TEST_DB_PORT,
                user="ragproject",
                password="ragproject",
                dbname="postgres",
                connect_timeout=3,
            )
            return conn
        except Exception as exc:  # noqa: BLE001 - retry any startup failure
            last_exc = exc
            time.sleep(1)
    raise RuntimeError(f"Test PostgreSQL never became ready: {last_exc}")


def _ensure_test_database() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the integration test suite")

    existing = _docker("ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}")
    if CONTAINER_NAME in existing.stdout:
        _docker("start", CONTAINER_NAME)
    else:
        result = _docker(
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-e",
            "POSTGRES_USER=ragproject",
            "-e",
            "POSTGRES_PASSWORD=ragproject",
            "-e",
            "POSTGRES_DB=postgres",
            "-p",
            f"{TEST_DB_PORT}:5432",
            "pgvector/pgvector:pg16",
            timeout=600,  # first run may pull the image
        )
        if result.returncode != 0:
            raise RuntimeError(f"Could not start test PostgreSQL: {result.stderr}")

    admin = _connect_admin()
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    except Exception:  # noqa: BLE001, S110 - "already exists" is expected
        pass
    finally:
        admin.close()

    # Apply migrations (idempotent: already-applied revisions are no-ops).
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed:\n{result.stdout}\n{result.stderr}"
        )


@pytest.fixture(scope="session")
def database_ready():
    """Start a pgvector container and migrate the test database once."""
    _ensure_test_database()
    yield


@pytest.fixture()
def client(database_ready):
    """FastAPI TestClient with lifespan startup/shutdown (real app, real DB)."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db_conn():
    """Direct synchronous connection for row-level assertions."""
    import psycopg2

    conn = psycopg2.connect(TEST_SYNC_DB_URL)
    conn.autocommit = True
    yield conn
    conn.close()


# --- Shared helpers --------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_rate_limits():
    from app.core.rate_limit import reset_rate_limits

    reset_rate_limits()
    yield
    reset_rate_limits()


@pytest.fixture(autouse=True)
def _no_welcome_email(monkeypatch):
    # The placeholder task sleeps 2s; keep it out of the request path in tests.
    monkeypatch.setattr("app.api.user.send_welcome_email", lambda email: None)


@pytest.fixture()
def account(client):
    """Register + login a fresh user; returns headers, email and id."""
    email = f"user-{uuid.uuid4().hex[:12]}@example.com"
    password = "sup3r-secret-pw"
    response = client.post("/users", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    user_id = response.json()["id"]

    response = client.post("/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]

    return {
        "headers": {"Authorization": f"Bearer {token}"},
        "email": email,
        "password": password,
        "user_id": user_id,
    }


async def fake_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Deterministic embeddings for upload tests (identical vectors → hit 0)."""
    return [[0.25] * TEST_EMBED_DIM for _ in texts]


@pytest.fixture()
def mock_embeddings(monkeypatch):
    monkeypatch.setattr("app.api.user.get_embeddings", fake_embeddings)


@pytest.fixture()
def mock_ask(monkeypatch):
    """Stub the external LLM + query embedding; keep real pgvector retrieval."""
    calls: dict[str, list] = {"prompts": []}

    async def fake_get_embedding(text, task_type="RETRIEVAL_QUERY"):
        return [0.25] * TEST_EMBED_DIM

    async def fake_stream(prompt):
        calls["prompts"].append(prompt)
        yield "Hello "
        yield "world"

    monkeypatch.setattr("app.services.rag_service.get_embedding", fake_get_embedding)
    monkeypatch.setattr("app.services.rag_service.stream_generate_answer", fake_stream)
    return calls


def make_pdf(text: str) -> bytes:
    """Generate a small, real PDF containing *text*."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # Keep lines reasonably short so extraction stays faithful.
    words = text.split()
    line, lines = "", []
    for word in words:
        if len(line) + len(word) + 1 > 90:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    page.insert_text((72, 72), "\n".join(lines) if lines else text)
    payload = doc.tobytes()
    doc.close()
    return payload
