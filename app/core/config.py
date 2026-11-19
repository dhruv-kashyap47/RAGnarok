import os

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    cleaned = value.strip()
    return cleaned or default


def get_bool_env(name: str, default: bool = False) -> bool:
    value = get_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def get_list_env(name: str, default: list[str] | None = None) -> list[str]:
    value = get_env(name)
    if value is None:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_int_env(name: str, default: int) -> int:
    value = get_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name} must be an integer, got {value!r}"
        ) from exc


def _get_float_env(name: str, default: float) -> float:
    value = get_env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name} must be a number, got {value!r}"
        ) from exc


DATABASE_URL = get_env(
    "DATABASE_URL",
    "postgresql+asyncpg://ragproject:ragproject@localhost:5432/ragproject",
)
REDIS_URL = get_env("REDIS_URL", "redis://localhost:6379")

SECRET_KEY = get_env("SECRET_KEY")
if not SECRET_KEY:
    # Fail fast: without a secret key every token mint/verify would break at
    # request time with an opaque 500. Never log or echo any secret value.
    raise RuntimeError(
        "SECRET_KEY environment variable is required. "
        'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
    )
ALGORITHM = get_env("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = _get_int_env("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
if ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
    raise RuntimeError("ACCESS_TOKEN_EXPIRE_MINUTES must be a positive integer")

SQL_ECHO = get_bool_env("SQL_ECHO", False)
PORT = _get_int_env("PORT", 8000)
CORS_ORIGINS = get_list_env("CORS_ORIGINS", ["*"])
ALLOW_CREDENTIALS = get_bool_env("ALLOW_CREDENTIALS", False)

# Hardening knobs -----------------------------------------------------------
# Maximum size of any request body (JSON, multipart, ...). Must comfortably
# exceed the PDF upload limit below plus multipart framing overhead.
MAX_REQUEST_BODY_BYTES = _get_int_env("MAX_REQUEST_BODY_BYTES", 25 * 1024 * 1024)
# Rate limits (requests per window, applied per key: see app.core.rate_limit).
AUTH_RATE_LIMIT_PER_MINUTE = _get_int_env("AUTH_RATE_LIMIT_PER_MINUTE", 30)
REGISTER_RATE_LIMIT_PER_MINUTE = _get_int_env("REGISTER_RATE_LIMIT_PER_MINUTE", 20)
ASK_RATE_LIMIT_PER_MINUTE = _get_int_env("ASK_RATE_LIMIT_PER_MINUTE", 30)
UPLOAD_RATE_LIMIT_PER_MINUTE = _get_int_env("UPLOAD_RATE_LIMIT_PER_MINUTE", 10)
# Set to true only when the app sits behind a trusted reverse proxy that
# sets X-Forwarded-For; otherwise clients could spoof their rate-limit key.
TRUST_PROXY_HEADERS = get_bool_env("TRUST_PROXY_HEADERS", False)
# Expose /docs, /redoc and /openapi.json (disable in internet-facing prod).
DOCS_ENABLED = get_bool_env("DOCS_ENABLED", True)

GEMINI_API_KEY = get_env("GEMINI_API_KEY")
GEMINI_API_BASE = get_env(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta/models",
)
GEMINI_CHAT_MODEL = get_env("GEMINI_CHAT_MODEL", "gemini-2.0-flash-001")
GEMINI_EMBED_MODEL = get_env("GEMINI_EMBED_MODEL", "gemini-embedding-001")

GROQ_API_KEY = get_env("GROQ_API_KEY")
GROQ_API_BASE = get_env("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_CHAT_MODEL = get_env("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
EMBEDDING_DIMENSIONS = _get_int_env("EMBEDDING_DIMENSIONS", 3072)

HTTP_TIMEOUT_SECONDS = _get_float_env("HTTP_TIMEOUT_SECONDS", 60)
CHAT_TIMEOUT_SECONDS = _get_float_env("CHAT_TIMEOUT_SECONDS", 120)
MAX_CHAT_HISTORY_MESSAGES = _get_int_env("MAX_CHAT_HISTORY_MESSAGES", 6)

if ALLOW_CREDENTIALS and "*" in CORS_ORIGINS:
    # Browsers reject `Access-Control-Allow-Origin: *` combined with
    # credentials, so this combination silently breaks cross-origin credentialed
    # requests. Warn loudly instead of shipping a confusing misconfiguration.
    import logging

    logging.getLogger("ragproject").warning(
        "ALLOW_CREDENTIALS=true with CORS_ORIGINS='*' is an invalid "
        "combination: browsers will reject credentialed cross-origin requests. "
        "List explicit origins in CORS_ORIGINS instead."
    )
