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
