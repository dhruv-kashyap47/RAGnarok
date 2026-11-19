import hmac
from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.exc import UnknownHashError

from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# NOTE: passlib guards lazy backend initialization with its own
# `_backend_lock`, so first-use hashing from worker threads is safe.


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except UnknownHashError:
        # Legacy rows may store a non-bcrypt value. Compare in constant time
        # so login timing does not reveal whether an account exists.
        return hmac.compare_digest(plain_password.encode(), hashed_password.encode())


def password_needs_rehash(stored_password: str) -> bool:
    try:
        return pwd_context.needs_update(stored_password)
    except UnknownHashError:
        return True


def create_access_token(
    data: dict, expires_delta: timedelta | None = None
) -> str:
    """Sign *data* into a JWT.

    ``expires_delta`` overrides the default ACCESS_TOKEN_EXPIRE_MINUTES
    lifetime (used for long-lived guest session tokens).
    """
    to_encode = data.copy()
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(UTC) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_access_token(token: str) -> dict | None:
    try:
        # Require exp so a token without an expiry is never accepted.
        # NOTE: python-jose uses the boolean `require_exp` option (not
        # PyJWT's `require: [...]` list, which jose silently ignores).
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require_exp": True},
        )
        user_id = payload.get("user_id")
        if user_id is None:
            return None
        return {"user_id": user_id, "role": payload.get("role")}
    except JWTError:
        return None
