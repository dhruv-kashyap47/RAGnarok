"""API tests: registration, login, token auth, and the user-listing lockdown."""

import uuid

from jose import jwt

from app.core.config import ALGORITHM, SECRET_KEY


def test_register_login_me_flow(client):
    email = f"flow-{uuid.uuid4().hex[:10]}@example.com"

    created = client.post("/users", json={"email": email, "password": "secret123"})
    assert created.status_code == 200
    body = created.json()
    assert body["email"] == email
    assert body["role"] == "user"
    assert "password" not in body

    login = client.post("/login", json={"email": email, "password": "secret123"})
    assert login.status_code == 200
    token = login.json()["access_token"]
    assert login.json()["token_type"] == "bearer"

    me = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email
    # The password hash must never leave the server.
    assert "password" not in me.json()


def test_duplicate_registration_is_400_not_500(client):
    email = f"dup-{uuid.uuid4().hex[:10]}@example.com"
    payload = {"email": email, "password": "secret123"}
    assert client.post("/users", json=payload).status_code == 200

    second = client.post("/users", json=payload)
    assert second.status_code == 400
    assert second.json()["message"] == "Email already exists"


def test_login_with_wrong_password(client):
    email = f"wrong-{uuid.uuid4().hex[:10]}@example.com"
    client.post("/users", json={"email": email, "password": "secret123"})

    response = client.post("/login", json={"email": email, "password": "nope-nope"})
    assert response.status_code == 400
    assert response.json()["message"] == "Invalid email or password"


def test_login_for_unknown_email_is_indistinguishable(client):
    response = client.post(
        "/login", json={"email": f"ghost-{uuid.uuid4().hex[:10]}@example.com", "password": "x"}
    )
    assert response.status_code == 400
    assert response.json()["message"] == "Invalid email or password"


def test_oauth2_token_endpoint(client):
    email = f"oauth-{uuid.uuid4().hex[:10]}@example.com"
    client.post("/users", json={"email": email, "password": "secret123"})

    response = client.post(
        "/token", data={"username": email, "password": "secret123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_register_validation(client):
    short = client.post("/users", json={"email": "a@b.com", "password": "123"})
    assert short.status_code == 422

    bad_email = client.post("/users", json={"email": "not-an-email", "password": "123456"})
    assert bad_email.status_code == 422


def test_login_validation(client):
    response = client.post("/login", json={"email": "not-an-email", "password": "x"})
    assert response.status_code == 422


def test_missing_token_returns_401_with_bearer_challenge(client):
    response = client.get("/me")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert response.json()["success"] is False


def test_invalid_token_returns_401_with_bearer_challenge(client):
    response = client.get("/me", headers={"Authorization": "Bearer not-a-token"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_token_with_non_uuid_user_id_is_rejected(client):
    """Regression: a signed token with a malformed user id must be rejected
    at the auth layer instead of causing type errors (500) downstream."""
    token = jwt.encode(
        {"user_id": "not-a-uuid", "role": "user", "exp": 4_102_444_800},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    response = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_expired_token_is_rejected(client):
    token = jwt.encode(
        {"user_id": str(uuid.uuid4()), "role": "user", "exp": 1_000_000_000},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    response = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_user_listing_requires_authentication(client):
    """Regression: GET /users used to expose every registered email
    without any authentication."""
    response = client.get("/users")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_user_listing_returns_safe_fields(client, account):
    # Use the server-side email filter: the shared test database accumulates
    # users across runs, so a default first page cannot be guaranteed to
    # contain this account.
    response = client.get(
        "/users", params={"email": account["email"]}, headers=account["headers"]
    )
    assert response.status_code == 200
    users = response.json()
    assert len(users) == 1
    assert users[0]["email"] == account["email"]
    for user in users:
        assert set(user) == {"id", "email", "role"}
        assert "password" not in user


def test_user_listing_pagination_bounds(client, account):
    headers = account["headers"]
    assert client.get("/users?skip=-1", headers=headers).status_code == 422
    assert client.get("/users?limit=0", headers=headers).status_code == 422
    assert client.get("/users?limit=1000", headers=headers).status_code == 422
    assert client.get("/users?limit=10&skip=0", headers=headers).status_code == 200


def test_email_is_normalized_on_register_and_login(client):
    email = f"mixed-{uuid.uuid4().hex[:8]}@example.com"
    created = client.post("/users", json={"email": email.upper(), "password": "secret123"})
    assert created.status_code == 200
    assert created.json()["email"] == email

    login = client.post("/login", json={"email": email.upper(), "password": "secret123"})
    assert login.status_code == 200


# --- Guest sessions ---------------------------------------------------------


def test_guest_session_mints_usable_token(client):
    response = client.post("/guest-session")
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["role"] == "guest"
    assert body["email"].startswith("guest-")
    assert body["user_id"]

    headers = {"Authorization": f"Bearer {body['access_token']}"}
    # Guests may use identity routes (ask/upload/documents)...
    assert client.get("/documents", headers=headers).status_code == 200
    # ...but never account-only routes.
    assert client.get("/me", headers=headers).status_code == 403
    assert client.get("/users", headers=headers).status_code == 403


def test_guest_sessions_are_distinct_users_with_bcrypt_secrets(
    client, db_conn
):
    first = client.post("/guest-session").json()
    second = client.post("/guest-session").json()
    assert first["user_id"] != second["user_id"]
    assert first["email"] != second["email"]

    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM users WHERE role = 'guest'")
        assert cur.fetchone()[0] == 2  # per-test TRUNCATE keeps counts exact
        # The guest password must be a bcrypt hash of a random secret —
        # never a shared sentinel (verify_password has a plaintext fallback).
        cur.execute("SELECT password FROM users WHERE id = %s", (first["user_id"],))
        stored = cur.fetchone()[0]
        assert stored.startswith("$2")


def _backdate_guest(db_conn, guest_id: str, days: int = 40) -> None:
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET created_at = now() - make_interval(days => %s) "
            "WHERE id = %s",
            (days, guest_id),
        )


def test_stale_docless_guests_are_reaped(client, db_conn):
    old = client.post("/guest-session").json()
    _backdate_guest(db_conn, old["user_id"])

    fresh = client.post("/guest-session").json()  # minting triggers the reap

    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM users WHERE id = %s", (old["user_id"],))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM users WHERE id = %s", (fresh["user_id"],))
        assert cur.fetchone()[0] == 1


def test_stale_guests_with_documents_survive_cleanup(
    client, guest_account, mock_embeddings, db_conn
):
    from tests.conftest import make_pdf

    uploaded = client.post(
        "/upload-pdf",
        headers=guest_account["headers"],
        files={"file": ("keep.pdf", make_pdf("keep me"), "application/pdf")},
    )
    assert uploaded.status_code == 200, uploaded.text
    _backdate_guest(db_conn, guest_account["user_id"])

    client.post("/guest-session")  # triggers the reap

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM users WHERE id = %s", (guest_account["user_id"],)
        )
        assert cur.fetchone()[0] == 1, "guests that own documents must not be reaped"
        cur.execute("SELECT count(*) FROM document_files")
        assert cur.fetchone()[0] == 1


def test_user_listing_excludes_guests(client, account, guest_account):
    response = client.get(
        "/users",
        params={"email": guest_account["email"]},
        headers=account["headers"],
    )
    assert response.status_code == 200
    assert response.json() == []


def test_invalid_guest_token_still_allows_registration(client):
    """X-Guest-Token is best-effort: garbage must not break auth."""
    email = f"garbage-{uuid.uuid4().hex[:8]}@example.com"
    created = client.post(
        "/users",
        json={"email": email, "password": "secret123"},
        headers={"X-Guest-Token": "not-a-jwt"},
    )
    assert created.status_code == 200


def test_user_token_in_guest_header_cannot_steal_documents(
    client, account, mock_embeddings, db_conn
):
    """Only role='guest' rows may be claimed: a registered user's token in
    X-Guest-Token must move nothing."""
    from tests.conftest import make_pdf

    client.post(
        "/upload-pdf",
        headers=account["headers"],
        files={"file": ("mine.pdf", make_pdf("owner content"), "application/pdf")},
    )

    email = f"thief-{uuid.uuid4().hex[:8]}@example.com"
    created = client.post(
        "/users",
        json={"email": email, "password": "secret123"},
        headers={"X-Guest-Token": account["headers"]["Authorization"].removeprefix("Bearer ")},
    )
    assert created.status_code == 200

    with db_conn.cursor() as cur:
        cur.execute("SELECT user_id FROM document_files")
        rows = cur.fetchall()
    assert len(rows) == 1
    assert str(rows[0][0]) == account["user_id"]  # untouched


def test_register_with_guest_token_transfers_documents(
    client, guest_account, mock_embeddings, db_conn
):
    from tests.conftest import make_pdf

    uploaded = client.post(
        "/upload-pdf",
        headers=guest_account["headers"],
        files={"file": ("guest.pdf", make_pdf("guest content"), "application/pdf")},
    )
    assert uploaded.status_code == 200, uploaded.text

    email = f"merge-{uuid.uuid4().hex[:8]}@example.com"
    created = client.post(
        "/users",
        json={"email": email, "password": "secret123"},
        headers={"X-Guest-Token": guest_account["guest_token"]},
    )
    assert created.status_code == 200, created.text
    new_user_id = created.json()["id"]

    with db_conn.cursor() as cur:
        cur.execute("SELECT user_id FROM document_files")
        file_owners = [str(r[0]) for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT user_id FROM documents")
        chunk_owners = [str(r[0]) for r in cur.fetchall()]
        # The guest row survives as a tombstone while its token is alive.
        cur.execute("SELECT count(*) FROM users WHERE role = 'guest'")
        guests = cur.fetchone()[0]

    assert file_owners == [new_user_id]
    assert chunk_owners == [new_user_id]
    assert guests == 1

    # The new account sees the merged document.
    token = client.post("/login", json={"email": email, "password": "secret123"}).json()[
        "access_token"
    ]
    listed = client.get(
        "/documents", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert [d["filename"] for d in listed] == ["guest.pdf"]


def test_login_with_guest_token_transfers_documents(
    client, guest_account, mock_embeddings, db_conn
):
    from tests.conftest import make_pdf

    uploaded = client.post(
        "/upload-pdf",
        headers=guest_account["headers"],
        files={"file": ("later.pdf", make_pdf("later content"), "application/pdf")},
    )
    assert uploaded.status_code == 200, uploaded.text

    # Register normally first (no guest header), then merge on login.
    email = f"later-{uuid.uuid4().hex[:8]}@example.com"
    assert (
        client.post("/users", json={"email": email, "password": "secret123"}).status_code
        == 200
    )
    login = client.post(
        "/login",
        json={"email": email, "password": "secret123"},
        headers={"X-Guest-Token": guest_account["guest_token"]},
    )
    assert login.status_code == 200, login.text

    with db_conn.cursor() as cur:
        cur.execute("SELECT user_id FROM document_files")
        owners = [str(r[0]) for r in cur.fetchall()]
    assert len(owners) == 1
    assert owners[0] != guest_account["user_id"]

    listed = client.get(
        "/documents", headers={"Authorization": f"Bearer {login.json()['access_token']}"}
    ).json()
    assert [d["filename"] for d in listed] == ["later.pdf"]
