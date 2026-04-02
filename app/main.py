from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import os

from app.core.config import ALLOW_CREDENTIALS, CORS_ORIGINS
from app.core.exceptions import http_exception_handler, general_exception_handler
from app.api.user import router as user_router
from app.models import user, documents

app = FastAPI(title="RAGnarok")

os.makedirs("static", exist_ok=True)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=ALLOW_CREDENTIALS,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ---------------- Static files ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/ui")
async def serve_ui():
    return FileResponse(
        "static/ragproject.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------------- Routers ----------------
app.include_router(user_router)

# ---------------- Exception handlers ----------------
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
    return {
        "status": "ok",
        "cors_origins": CORS_ORIGINS,
    }

@app.on_event("startup")
async def startup():
    # Schema changes are managed via Alembic migrations (see start.sh).
    # Keeping startup side-effect free avoids schema drift in production.
    return
