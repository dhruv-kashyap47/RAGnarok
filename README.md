# 🔥 RAGnarok: Your Personal Knowledge Engine

Welcome to **RAGnarok** — not just another RAG app, but a **precision-driven knowledge system** built to eliminate hallucinations and deliver grounded, context-aware answers.

Upload your documents. Turn them into intelligence. Query them like a god.

---

## ⚡ What This Does (Straight Up)

* No guessing. No fake answers.
* Every response is backed by **your actual data**
* Built for **speed, accuracy, and real-world usage**

---

## 🚀 Core Features

### 📂 Smart Document Ingestion

Upload PDFs → automatically parsed, chunked, and embedded into vectors.
Your raw data becomes searchable intelligence.

### ⚡ High-Speed Semantic Search

Powered by **PostgreSQL + pgvector**
Find meaning, not just keywords.

### 💬 Real-Time AI Chat (Streaming)

Ask questions → get **instant, context-aware answers**
Supports:

* Google Gemini
* Groq

### 🧠 True RAG Pipeline

Retrieve → Rank → Generate
Every answer is grounded in your documents.

### 🗂 Document Control

Upload, manage, and delete files easily — your data stays in your control.

---

## 🛠 Tech Stack (No BS)

* **Backend:** FastAPI, SQLAlchemy, HTTPX
* **Database:** PostgreSQL + pgvector
* **LLMs:** Gemini (default), Groq (optional boost mode)
* **Frontend:** Vanilla JS + HTML + modern CSS (fast, clean, no bloat)

---

## ⚙️ Setup (Do It Once, Done Forever)

### 1. Requirements

* Docker + Docker Compose
* Python 3.10+
* Gemini API Key
* (Optional) Groq API Key

---

### 2. Environment Setup

Create `.env`:

```env
SECRET_KEY=use-a-strong-random-64-char-string
GEMINI_API_KEY=your_key_here
GEMINI_CHAT_MODEL=gemini-2.0-flash-001
GROQ_API_KEY=
DATABASE_URL=postgresql+asyncpg://ragproject:ragproject@localhost:5432/ragproject
REDIS_URL=redis://localhost:6379
SQL_ECHO=False
```

---

### 3. Spin Up Database

```bash
docker-compose up -d
```

Apply schema:

```bash
alembic upgrade head
```

---

### 4. Run Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

```
http://localhost:8000
```

---

## 🧠 How It Actually Works

This is where most people get confused — here’s the clean breakdown:

### 1. Input Layer (Frontend)

Simple UI → sends requests to backend

### 2. Processing Layer (Backend API)

Handles:

* auth
* uploads
* chat queries

### 3. Intelligence Layer (Core RAG Engine)

* **PDF → Text extraction**
* **Text → Chunks**
* **Chunks → Embeddings (vectors)**
* **Vectors → Stored in DB**
* **Query → Similar chunks retrieved**
* **LLM → Generates answer using ONLY retrieved context**

👉 That’s why it doesn’t hallucinate like dumb chatbots.

---

## 🔐 Security & Safety

* Your data stays **isolated per user**
* No external data leakage
* API keys stored via environment variables
* No hidden tracking, no shady stuff

---

## 💡 Why This Matters

Most AI apps:

* Guess
* Hallucinate
* Sound confident but wrong

**RAGnarok doesn’t play that game.**

It:

* Retrieves real data
* Grounds every response
* Delivers answers you can trust

---

## 🧨 Final Take

This isn’t just a project.

It’s:

* A **production-ready backend system**
* A **resume killer project**
* A **foundation for real AI products**
