"""Smoke tests for app wiring: health, static UI, CORS, error shapes."""


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "cors_origins" in body


def test_api_root(client):
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json() == {"message": "Backend running successfully"}


def test_root_redirects_to_ui(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 303, 307, 308)
    assert response.headers["location"] == "/ui"


def test_ui_served_with_no_cache(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "no-store" in response.headers.get("cache-control", "")


def test_static_files_served(client):
    response = client.get("/static/ragproject.html")
    assert response.status_code == 200


def test_static_and_ui_are_cwd_independent(client, monkeypatch, tmp_path):
    """Regression: static paths were resolved relative to the process cwd,
    so running uvicorn from another directory broke /ui and /static."""
    monkeypatch.chdir(tmp_path)
    assert client.get("/ui").status_code == 200
    assert client.get("/static/ragproject.html").status_code == 200


def test_unknown_route_uses_unified_error_shape(client):
    # Starlette-level 404s (e.g. static misses) must share the API's
    # {"success", "message"} envelope with route-level HTTPExceptions.
    response = client.get("/definitely-not-a-route")
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["message"] == "Not Found"


def test_docs_enabled_by_default(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_cors_preflight_allows_configured_origins(client):
    response = client.options(
        "/login",
        headers={
            "Origin": "https://example.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"


def test_health_does_not_leak_secrets(client):
    body = client.get("/health").text
    assert "SECRET_KEY" not in body
    assert "GROQ_API_KEY" not in body
    assert "gsk_" not in body


def test_capabilities_reports_groq_and_768_dimensions(client):
    from app.core.config import EMBEDDING_DIMENSIONS, GROQ_CHAT_MODEL

    body = client.get("/capabilities").json()
    assert body["llm_provider"] in {"groq", "unconfigured"}  # never another vendor
    assert "gemini" not in str(body).lower()
    assert body["llm_model"] == GROQ_CHAT_MODEL
    assert body["embedding_provider"] == "groq"
    assert body["embedding_dimensions"] == EMBEDDING_DIMENSIONS == 768
