# ---- Base Image ----
FROM python:3.11-slim

# ---- Env Settings ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System Dependencies (important for psycopg, pdf, etc.) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Install Python Dependencies ----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Copy App ----
COPY . .

# ---- Expose Port ----
EXPOSE 10000

# ---- Start Command (Render-friendly + DB-safe) ----
CMD ["sh", "-c", "python -m app.init_db && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
