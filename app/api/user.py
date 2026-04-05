from fastapi import APIRouter, Depends, BackgroundTasks, Query, UploadFile, File, HTTPException
import os
import tempfile

from app.schemas.user import UserCreate, UserLogin
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.user_service import UserService
from app.core.dependencies import get_current_user, get_user_service
from app.core.tasks import send_welcome_email
from app.services.embedding_service import (
    get_embedding_dimensions,
    get_embedding_provider,
    get_embeddings,
)
from app.services.llm_service import get_llm_model, get_llm_provider
from app.services.vector_store import (
    create_document_record,
    list_documents,
    store_documents_bulk,
    delete_document,
)
from app.services.rag_service import ask_question_stream
from app.services.chunking import chunk_text
from app.services.pdf_service import extract_text_from_pdf
from app.db.database import get_db

router = APIRouter()


@router.post("/users")
async def create_user(
    user: UserCreate,
    background_tasks: BackgroundTasks,
    service: UserService = Depends(get_user_service),
):
    result = await service.create_user(user.email, user.password)
    background_tasks.add_task(send_welcome_email, user.email)
    return result


@router.post("/login")
async def login_user(user: UserLogin, service: UserService = Depends(get_user_service)):
    return await service.login_user(user.email, user.password)


from fastapi.security import OAuth2PasswordRequestForm

@router.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    service: UserService = Depends(get_user_service),
):
    return await service.login_user(form_data.username, form_data.password)


@router.get("/me")
async def get_me(
    current_user: dict = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
):
    return await service.get_user_by_id(current_user["user_id"])


@router.get("/users")
async def get_users(
    skip: int = Query(0),
    limit: int = Query(10),
    email: str = Query(None),
    service: UserService = Depends(get_user_service),
):
    return await service.get_users(skip, limit, email)


@router.get("/documents")
async def get_documents(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    return await list_documents(db, current_user["user_id"])


@router.delete("/documents/{document_id}")
async def delete_user_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    success = await delete_document(db, current_user["user_id"], document_id)
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
    question: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        ask_question_stream(db, question, current_user["user_id"], history=[]),
        media_type="application/x-ndjson"
    )


@router.post("/ask")
async def ask_with_history(
    chat: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        ask_question_stream(
            db,
            chat.question,
            current_user["user_id"],
            history=[item.model_dump() for item in chat.history],
        ),
        media_type="application/x-ndjson"
    )


_MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB hard limit


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    from app.core.logger import logger

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
        # --- Read & size-check in memory before writing to disk ---
        raw_bytes = await file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(raw_bytes) > _MAX_PDF_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds the {_MAX_PDF_BYTES // (1024 * 1024)} MB size limit.",
            )

        # --- Write to temp file ---
        suffix = os.path.splitext(filename)[1] or ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        file_path = tmp.name
        try:
            tmp.write(raw_bytes)
        finally:
            tmp.close()

        del raw_bytes  # free memory before embedding

        # --- Extract text ---
        logger.info(
            "upload-pdf: extracting text from '%s' (user_id=%s)",
            filename,
            current_user["user_id"],
        )
        text = extract_text_from_pdf(file_path).strip()
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

        # --- Embed ---
        embeddings = await get_embeddings(chunks)

        # --- Persist ---
        document_record = await create_document_record(db, current_user["user_id"], filename)
        await store_documents_bulk(
            db,
            current_user["user_id"],
            document_record.id,
            list(zip(chunks, embeddings)),
        )
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
