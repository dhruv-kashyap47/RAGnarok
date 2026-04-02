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


DATABASE_URL = get_env(
    "DATABASE_URL",
    "postgresql+asyncpg://ragproject:ragproject@localhost:5432/ragproject",
)
REDIS_URL = get_env("REDIS_URL", "redis://localhost:6379")

SECRET_KEY = get_env("SECRET_KEY")
ALGORITHM = get_env("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(get_env("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

SQL_ECHO = get_bool_env("SQL_ECHO", False)
PORT = int(get_env("PORT", "8000"))
CORS_ORIGINS = get_list_env("CORS_ORIGINS", ["*"])
ALLOW_CREDENTIALS = get_bool_env("ALLOW_CREDENTIALS", False)

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
EMBEDDING_DIMENSIONS = int(get_env("EMBEDDING_DIMENSIONS", "768"))

HTTP_TIMEOUT_SECONDS = float(get_env("HTTP_TIMEOUT_SECONDS", "60"))
CHAT_TIMEOUT_SECONDS = float(get_env("CHAT_TIMEOUT_SECONDS", "120"))
MAX_CHAT_HISTORY_MESSAGES = int(get_env("MAX_CHAT_HISTORY_MESSAGES", "6"))
