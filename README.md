# RAGProject: Your Personal Knowledge Engine 🚀

Welcome to **RAGProject** - a modern, mobile-responsive, production-grade Retrieval-Augmented Generation (RAG) platform. 

This project allows you to upload PDFs, convert them into vector embeddings, and interactively chat with your actual document content using Gemini or Groq-backed generation. It's built to be robust but understandable for beginners!

## 🌟 Key Features

- **Upload & Embed**: Securely upload PDF documents which are parsed, chunked, and saved to a high-speed database.
- **Lightning-Fast Vector Search**: Uses PostgreSQL's `pgvector` to do blazing-fast semantic searches on your documents.
- **Live Streaming Chat**: The LLM uses your knowledge base to accurately answer your questions in real-time! No more waiting for complete generation!
- **Document Management**: Manage your workspace natively, including deleting older documents.

## 🛠 Tech Stack

- **Backend:** FastAPI, SQLAlchemy, HTTPX
- **Database:** PostgreSQL with `pgvector`
- **LLM/Embeddings:** Google Gemini, with optional Groq for chat generation
- **Frontend:** Vanilla JS / HTML / Modern CSS (Zero bulky frameworks!)

## 🚀 Quick Setup Guide

### 1. Prerequisites
- Docker and Docker Compose (to run the database)
- Python 3.10+
- A Google Gemini API Key
- Optional: a Groq API key if you want Groq for chat generation instead of Gemini

### 2. Environment Variables
Create a file named `.env` in the root of the project. Gemini is enough for both embeddings and chat. If `GROQ_API_KEY` is also set, chat generation will prefer Groq.
```env
SECRET_KEY=change-me-to-a-random-64-char-hex-string
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-2.0-flash-001
GROQ_API_KEY=
DATABASE_URL=postgresql+asyncpg://ragproject:ragproject@localhost:5432/ragproject
REDIS_URL=redis://localhost:6379
SQL_ECHO=False
```

### 3. Database Bootstrapping
You don't need to install Postgres! We provide a `docker-compose.yml` that will boot up Postgres equipped with `pgvector`.
```bash
docker-compose up -d
```
Then, apply the database schema (migrations) through Alembic:
```bash
alembic upgrade head
```

### 4. Running the Server
Make sure you've installed the Python dependencies:
```bash
pip install -r requirements.txt
```
Run the FastAPI development server:
```bash
uvicorn app.main:app --reload
```

Then visit [http://localhost:8000](http://localhost:8000) in your browser!

---

## 📖 Architecture Deep Dive (For Beginners)

Here's how requests trace through the RAG pipeline:
1. **Frontend (`static/ragproject.html`)**: A lightweight UI that directly connects to our backend APIs.
2. **REST API (`app/api/`)**: Built with FastAPI. This handles authentication, uploading documents, and receiving chat messages.
3. **Services (`app/services/`)**: The core brain. 
   - `pdf_service.py` extracts text from PDFs.
   - `chunking.py` splits text into smaller, readable pieces.
   - `embedding_service.py` asks Gemini for vector representations (arrays of numbers) of the text.
   - `vector_store.py` stores these numbers and text into Postgres.
   - `rag_service.py` combines semantic search (fetching chunks of your PDF most similar to your query) with the configured chat model via a streaming response.
4. **Database Models (`app/models/`)**: Defines how pieces of data map to PostgreSQL tables.

Enjoy using RAGProject!
