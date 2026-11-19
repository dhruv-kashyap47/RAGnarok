import os
import tempfile
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.config import (
    ASK_RATE_LIMIT_PER_MINUTE,
    AUTH_RATE_LIMIT_PER_MINUTE,
    REGISTER_RATE_LIMIT_PER_MINUTE,
    UPLOAD_RATE_LIMIT_PER_MINUTE,
)
from app.core.dependencies import get_current_user, get_user_service
from app.core.rate_limit import auth_bucket, enforce_rate_limit
from app.core.tasks import send_welcome_email
from app.db.database import get_db
from app.schemas.chat import ChatRequest
from app.schemas.user import UserCreate, UserLogin
from app.services.chunking import chunk_text
from app.services.embedding_service import (
    get_embedding_dimensions,
    get_embedding_provider,
    get_embeddings,
)
from app.services.llm_service import get_llm_model, get_llm_provider
from app.services.pdf_service import extract_text_from_pdf
from app.services.rag_service import ask_question_stream
from app.services.user_service import UserService
from app.services.vector_store import (
    create_document_record,
    delete_document,
    list_documents,
    store_documents_bulk,
)

router = APIRouter()

# Embedding API accepts at most 100 texts per batch call.
_EMBED_BATCH_SIZE = 100
_MAX_QUESTION_LENGTH = 4000  # keep in sync with schemas.chat.ChatRequest


@router.post("/users")
async def create_user(
    user: UserCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    service: UserService = Depends(get_user_service),
):
    enforce_rate_limit(
        request,
        auth_bucket(request, "register"),
        REGISTER_RATE_LIMIT_PER_MINUTE,
    )
    result = await service.create_user(user.email, user.password)
    background_tasks.add_task(send_welcome_email, user.email)
    return result


@router.post("/login")
async def login_user(
    user: UserLogin,
    request: Request,
    service: UserService = Depends(get_user_service),
):
    enforce_rate_limit(
        request,
        auth_bucket(request, "login", user.email),
        AUTH_RATE_LIMIT_PER_MINUTE,
    )
    return await service.login_user(user.email, user.password)


@router.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
    service: UserService = Depends(get_user_service),
):
    enforce_rate_limit(
        request,
        auth_bucket(request, "login", form_data.username),
        AUTH_RATE_LIMIT_PER_MINUTE,
    )
    return await service.login_user(form_data.username, form_data.password)


@router.get("/me")
async def get_me(
    current_user: dict = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
):
    return await service.get_user_by_id(current_user["user_id"])


@router.get("/users")
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    email: str | None = Query(None),
    current_user: dict = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
):
    # Listing accounts exposes every registered email: authenticated users only.
    return await service.get_users(skip, limit, email)


@router.get("/documents")
async def get_documents(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await list_documents(db, current_user["user_id"])


@router.delete("/documents/{document_id}")
async def delete_user_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # A malformed id must not reach the DB driver (it would surface as a 500).
    try:
        document_uuid = UUID(document_id)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail="Document not found")

    success = await delete_document(db, current_user["user_id"], document_uuid)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    await db.commit()
    return {"message": "Document deleted successfully"}


@router.get("/capabilities")
async def get_capabilities():
    return {
        "llm_provider": get_llm_provider(),
        "llm_model": get_llm_model(),
        "embedding_provider": get_embedding_provider(),
        "embedding_dimensions": get_embedding_dimensions(),
    }


@router.get("/ask")
async def ask(
    question: str = Query(..., max_length=_MAX_QUESTION_LENGTH),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    # Validate before creating the streaming response: raising after the
    # stream has started cannot become a clean 400 anymore.
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    enforce_rate_limit(
        request,
        f"ask:{current_user['user_id']}",
        ASK_RATE_LIMIT_PER_MINUTE,
    )
    return StreamingResponse(
        ask_question_stream(db, question, current_user["user_id"], history=[]),
        media_type="application/x-ndjson",
    )


@router.post("/ask")
async def ask_with_history(
    chat: ChatRequest,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    enforce_rate_limit(
        request,
        f"ask:{current_user['user_id']}",
        ASK_RATE_LIMIT_PER_MINUTE,
    )
    return StreamingResponse(
        ask_question_stream(
            db,
            chat.question,
            current_user["user_id"],
            history=[item.model_dump() for item in chat.history],
        ),
        media_type="application/x-ndjson",
    )


_MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB hard limit


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    from app.core.logger import logger

    enforce_rate_limit(
        request,
        f"upload:{current_user['user_id']}",
        UPLOAD_RATE_LIMIT_PER_MINUTE,
    )

    filename = (file.filename or "document.pdf").strip()

    # --- Validate filename extension ---
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # --- Validate MIME type if provided ---
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ("application/pdf", "application/octet-stream", ""):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{content_type}'. Expected application/pdf.",
        )

    # Safe default; will be set before the finally block is reached
    file_path: str | None = None
    document_record = None
    chunks: list[str] = []

    try:
        # --- Stream to disk with a hard size cap ---
        # Reading the whole body into memory first would let a huge (or
        # lying) upload exhaust RAM before the size check ever runs.
        suffix = os.path.splitext(filename)[1] or ".pdf"
        # delete=False is intentional: the file must outlive this handle for
        # PDF extraction, and is removed in the `finally` block below.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)  # noqa: SIM115
        file_path = tmp.name
        size = 0
        try:
            while True:
                piece = await file.read(1024 * 1024)
                if not piece:
                    break
                size += len(piece)
                if size > _MAX_PDF_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"PDF exceeds the {_MAX_PDF_BYTES // (1024 * 1024)} MB size limit."
                        ),
                    )
                tmp.write(piece)
        finally:
            tmp.close()

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # --- Extract text (CPU-bound: run off the event loop) ---
        logger.info(
            "upload-pdf: extracting text from '%s' (user_id=%s)",
            filename,
            current_user["user_id"],
        )
        text = (await run_in_threadpool(extract_text_from_pdf, file_path)).strip()
        if not text:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in this PDF. It may be a scanned image.",
            )

        # --- Chunk ---
        chunks = [c for c in chunk_text(text) if c.strip()]
        if not chunks:
            raise HTTPException(status_code=400, detail="Unable to create chunks from this PDF.")

        logger.info(
            "upload-pdf: produced %d chunks for '%s' (user_id=%s)",
            len(chunks),
            filename,
            current_user["user_id"],
        )

        # --- Persist header first (rolled back on any later failure) ---
        document_record = await create_document_record(
            db, current_user["user_id"], filename
        )

        # --- Embed + insert in bounded batches ---
        # A single giant embed/insert would hold every vector in memory at
        # once and could exceed PostgreSQL's bind-parameter limit.
        for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            embeddings = await get_embeddings(batch)
            if len(embeddings) != len(batch):
                raise HTTPException(
                    status_code=502,
                    detail="Embedding service returned an unexpected number of vectors.",
                )
            await store_documents_bulk(
                db,
                current_user["user_id"],
                document_record.id,
                list(zip(batch, embeddings)),
            )
            del embeddings

        await db.commit()

        logger.info(
            "upload-pdf: committed document_id=%s '%s' (%d chunks) for user_id=%s",
            document_record.id,
            filename,
            len(chunks),
            current_user["user_id"],
        )

    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        logger.error(
            "upload-pdf: unexpected error for '%s' (user_id=%s): %s",
            filename,
            current_user["user_id"],
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the PDF.",
        ) from exc
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

    return {
        "message": "PDF processed successfully",
        "document_id": str(document_record.id),
        "filename": document_record.filename,
        "chunks": len(chunks),
    }
