"""Unit tests for password hashing and JWT handling."""

import uuid

from jose import jwt

from app.core import security
from app.core.config import ALGORITHM, SECRET_KEY


def test_password_hash_roundtrip():
    stored = security.hash_password("hunter22")
    assert stored != "hunter22"
    assert stored.startswith("$2")
    assert security.verify_password("hunter22", stored)
    assert not security.verify_password("wrong-password", stored)


def test_verify_password_accepts_legacy_plaintext_value():
    # Legacy rows may hold non-bcrypt values; they must still verify
    # correctly (constant-time compare) instead of raising.
    assert security.verify_password("legacy-pass", "legacy-pass")
    assert not security.verify_password("nope", "legacy-pass")


def test_password_needs_rehash_flags_legacy_values():
    assert security.password_needs_rehash("legacy-pass") is True
    assert security.password_needs_rehash(security.hash_password("abcdef")) is False


def test_access_token_roundtrip():
    user_id = str(uuid.uuid4())
    token = security.create_access_token({"user_id": user_id, "role": "user"})
    payload = security.verify_access_token(token)
    assert payload == {"user_id": user_id, "role": "user"}


def test_token_without_exp_is_rejected():
    # Tokens without an expiry must never be accepted, even with a valid key.
    token = jwt.encode({"user_id": str(uuid.uuid4())}, SECRET_KEY, algorithm=ALGORITHM)
    assert security.verify_access_token(token) is None


def test_token_with_wrong_key_is_rejected():
    token = jwt.encode(
        {"user_id": str(uuid.uuid4()), "exp": 9_999_999_999},
        "attacker-key",
        algorithm=ALGORITHM,
    )
    assert security.verify_access_token(token) is None


def test_tampered_token_is_rejected():
    token = security.create_access_token({"user_id": str(uuid.uuid4())})
    tampered = token[:-2] + ("AA" if token[-2:] != "AA" else "BB")
    assert security.verify_access_token(tampered) is None


def test_garbage_token_is_rejected():
    assert security.verify_access_token("not-a-jwt") is None
    assert security.verify_access_token("") is None


def test_token_without_user_id_is_rejected():
    token = security.create_access_token({"role": "admin"})
    assert security.verify_access_token(token) is None
