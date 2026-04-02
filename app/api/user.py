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


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    suffix = os.path.splitext(filename)[1] or ".pdf"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_path = temp_file.name
    chunks = []

    try:
        temp_file.write(await file.read())
        temp_file.close()

        text = extract_text_from_pdf(file_path).strip()
        if not text:
            raise HTTPException(status_code=400, detail="No extractable text found in this PDF")

        chunks = [chunk for chunk in chunk_text(text) if chunk.strip()]
        if not chunks:
            raise HTTPException(status_code=400, detail="Unable to create chunks from this PDF")

        embeddings = await get_embeddings(chunks)

        document_record = await create_document_record(db, current_user["user_id"], filename)
        await store_documents_bulk(db, current_user["user_id"], document_record.id, list(zip(chunks, embeddings)))
        await db.commit()

    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        await db.rollback()
        raise
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "message": "PDF processed successfully",
        "document_id": str(document_record.id),
        "filename": document_record.filename,
        "chunks": len(chunks),
    }
