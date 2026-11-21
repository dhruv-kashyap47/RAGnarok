# RAGnarok: Your Personal Knowledge Engine

RAGnarok is a retrieval-augmented generation app for asking questions about your
own PDFs. It parses and chunks uploads, embeds them into PostgreSQL with
pgvector, retrieves the most relevant passages, and streams answers grounded in
those passages.

Upload your documents, build a private library, and query it in plain language.

---

## What It Does

- Answers questions from your documents instead of guessing.
- Returns the sources used for each answer.
- Supports guest use before sign-up, then merges guest uploads into an account.
- Streams answers so the first tokens appear immediately.
- Keeps conversation history per user or guest in the browser.

---

## Interface

The frontend is a single-file app in `static/ragproject.html`, served at `/ui`.
It is built around a focused, three-zone desktop workspace:

- **Icon rail** for new chats, conversations, documents, activity, API docs, and
  account access.
- **Conversation library** with search, recent history, and delete with undo.
- **Centered chat stage** with suggestion chips, a pill composer, streamed
  answers, source chips, and copy actions.

Supporting surfaces include:

- A white **Documents** sheet with drag and drop uploads, PDF validation, file
  metadata, and inline delete confirmation.
- A right-side **Activity** drawer with backend health, active models, vector
  dimensions, session state, and developer links.
- A focused **auth screen** for sign in, account creation, or guest access.

Convenience and accessibility details:

- Enter sends, Shift + Enter adds a new line.
- Ctrl/Cmd + K focuses conversation search.
- Escape closes open panels and dialogs.
- Stop generation without losing the partial answer.
- Markdown answers support headings, lists, links, tables, inline code, and code
  blocks.
- Responsive layouts keep the composer reachable on phones and tablets.
- System fonts only. No external font or asset requests.

---

## Core Features

### Smart Document Ingestion

Upload PDFs and RAGnarok parses, chunks, and embeds them automatically. The
Documents sheet validates file type and size before upload and confirms how many
searchable chunks were created.

### High-Speed Semantic Search

Powered by PostgreSQL and pgvector with an HNSW index. Retrieval ranks passages
by meaning rather than keyword overlap.

### Real-Time AI Chat

Answers stream into the chat as they are generated. The interface marks whether
retrieval was used, when no matching passages were found, and when generation was
stopped.

### True RAG Pipeline

Retrieve, rank, then generate. Answers are built from retrieved context and the
UI shows the source documents for every grounded response.

### Conversation Control

Search conversations, switch between them, copy answers, and delete chats with
an Undo action. History is stored in local storage under the current user or
guest identity.

### Guest-First Access

Anyone can ask questions and upload PDFs before registering. Signing in or
creating an account merges the guest library and conversations into that account
automatically.

---

## Tech Stack

- **Backend:** FastAPI, SQLAlchemy, HTTPX
- **Database:** PostgreSQL + pgvector with an HNSW index
- **Cache and limits:** Redis
- **LLM:** Groq `llama-3.1-8b-instant` on the free developer tier
- **Embeddings:** Gemini `gemini-embedding-001`, requested at 768 dimensions.
  Groq is chat-only because its `nomic-embed-text-v1_5` embedding model returns
  404 for normal API keys.
- **Frontend:** Semantic HTML, modern CSS, and vanilla JavaScript in one file,
  with no build step and no external assets.

---

## Setup

### 1. Requirements

- Docker and Docker Compose
- Python 3.10 or newer
- A Groq API key for chat
- A Gemini API key for embeddings

### 2. Environment

Create `.env` in the repository root:

```env
SECRET_KEY=use-a-strong-random-64-char-string
GROQ_API_KEY=your_key_here
GROQ_CHAT_MODEL=llama-3.1-8b-instant
GEMINI_API_KEY=your_key_here
GEMINI_EMBED_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768
DATABASE_URL=postgresql+asyncpg://ragproject:ragproject@localhost:5432/ragproject
REDIS_URL=redis://localhost:6379
SQL_ECHO=False
```

`SECRET_KEY`, API keys, and the database URL are read from the server
environment. Never commit `.env`; it is ignored by Git.

Changing `GEMINI_EMBED_MODEL` or `EMBEDDING_DIMENSIONS` requires a migration that
wipes stored vectors and resizes the vector column. `GET /health` reports the
embedding provider self-check result.

### 3. Start the Database

```bash
docker-compose up -d
```

Apply the schema:

```bash
alembic upgrade head
```

### 4. Run the Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/ui
```

The root path redirects to `/ui`. Interactive API documentation is available at
`/docs` when docs are enabled.

---

## Usage

1. Open `/ui`. A guest session starts automatically.
2. Upload a PDF from the composer paperclip, the Documents sheet, or by dragging
   it onto the upload area.
3. Wait for indexing to finish, then ask a question in plain language.
4. Review the answer and its source chips. Use Copy to reuse the answer.
5. Create an account when you want to keep the library tied to an email and
   password.

Guest uploads and conversations are merged into the account on sign-up or sign-in.

---

## API Overview

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/users` | Create an account |
| `POST` | `/login` | Sign in and receive a token |
| `POST` | `/guest-session` | Create a guest identity |
| `GET` | `/me` | Read the current profile |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/upload-pdf` | Upload, parse, chunk, and embed a PDF |
| `DELETE` | `/documents/{id}` | Delete a document |
| `POST` | `/ask` | Stream a grounded answer as NDJSON |
| `GET` | `/capabilities` | Read active provider and model details |
| `GET` | `/health` | Read backend and embedding health |

Authenticated requests use `Authorization: Bearer <token>`. Guest upgrades may
send `X-Guest-Token` when creating or signing into an account.

The `/ask` response is newline-delimited JSON. It starts with a metadata packet
containing sources, retrieval status, and warnings, followed by chunk packets
containing answer text.

---

## How It Works

### 1. Input Layer

The browser sends auth, upload, and chat requests to the FastAPI backend. The
frontend keeps conversations in local storage and never stores provider keys.

### 2. Processing Layer

The backend handles authentication, upload limits, PDF extraction, chunking,
embedding, retrieval, and streaming chat.

### 3. Intelligence Layer

- PDF to text extraction
- Text to overlapping chunks
- Chunks to embeddings
- Vectors stored in PostgreSQL with pgvector
- Query embedding and similarity search
- LLM generation constrained to retrieved context
- Metadata streamed before answer chunks

---

## Security and Safety

- Documents are scoped to the authenticated user or guest session.
- Provider API keys stay on the server and are never sent to the browser.
- `.env` and local runtime data are ignored by Git.
- Uploads are validated and subject to server request body limits.
- Markdown is escaped before formatting, and raw HTML from answers is not
  executed.
- Responses are marked when retrieval is unavailable, no passages matched, or
  generation was stopped.
- Health polling surfaces backend and embedding failures without exposing
  secrets.
- Session expiry falls back to a fresh guest session instead of silently failing.

---

## Testing

Run the suite with Docker running:

```bash
python -m pytest -q
```

The integration suite uses Docker to start disposable pgvector and Redis
containers. Install the Python requirements first so test collection can import
Redis and the application modules.

There is no frontend build step. `static/ragproject.html` is served directly by
FastAPI.

---

## Project Layout

```text
app/                 FastAPI application, API routes, services, models
alembic/             Database migrations
static/              Single-file frontend served at /ui
tests/               API, security, and integration tests
DESIGN.md            Visual design reference for the workspace
docker-compose.yml   PostgreSQL and Redis services
requirements.txt     Python dependencies
```

---

## Final Take

RAGnarok is a focused RAG workspace with a grounded backend and a fast,
dependency-free frontend. It is built to make document Q&A feel immediate while
keeping source visibility, guest access, and account migration simple.
