from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import RedirectResponse

from app.api.user import router as user_router
from app.core.config import (
    ALLOW_CREDENTIALS,
    CORS_ORIGINS,
    DOCS_ENABLED,
    MAX_REQUEST_BODY_BYTES,
)
from app.core.exceptions import (
    general_exception_handler,
    http_exception_handler,
)
from app.core.middleware import BodySizeLimitMiddleware
from app.models import documents, user  # noqa: F401  (registers ORM metadata)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: schema changes are managed via Alembic migrations (see
    # start.sh), so startup stays side-effect free to avoid schema drift.
    # The embedding provider is probed once so a bad key/model shows up in
    # the logs and /health instead of surfacing on the first user upload.
    from app.services.embedding_service import probe_embeddings

    await probe_embeddings()
    yield
    # Shutdown: release shared external resources cleanly.
    from app.core.redis import close_redis
    from app.db.database import engine
    from app.services.embedding_service import close_http_client as close_embed_http
    from app.services.llm_service import close_http_client as close_llm_http

    await close_embed_http()
    await close_llm_http()
    await close_redis()
    await engine.dispose()


app = FastAPI(
    title="RAGnarok",
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
    lifespan=lifespan,
)

STATIC_DIR.mkdir(exist_ok=True)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=ALLOW_CREDENTIALS,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
# Outermost: refuse oversized bodies before anything buffers them.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)

# ---------------- Static files ----------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/ui")
async def serve_ui():
    return FileResponse(
        STATIC_DIR / "ragproject.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------------- Routers ----------------
app.include_router(user_router)

# ---------------- Exception handlers ----------------
# Registering against Starlette's HTTPException also covers FastAPI's subclass
# (handler lookup walks the MRO), so static-file 404s and route errors share
# one response shape and auth challenge headers survive.
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# ---------------- Root ----------------
@app.get("/")
async def root():
    return RedirectResponse(url="/ui")


@app.get("/api")
async def api_root():
    return {"message": "Backend running successfully"}


# ---------------- Health ----------------
@app.get("/health")
async def health():
    # Always 200 so platform health checks keep the service alive even when a
    # provider is down; the payload carries the diagnostic instead.
    from app.services.embedding_service import get_embedding_health

    embeddings = get_embedding_health()
    return {
        "status": "ok",
        "degraded": embeddings["ok"] is False,
        "cors_origins": CORS_ORIGINS,
        "embeddings": embeddings,
    }
