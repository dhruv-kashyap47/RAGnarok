# ---- Base Image ----
FROM python:3.11-slim

# ---- Env Settings ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System Dependencies ----
# libmupdf-dev / mupdf-tools are needed by PyMuPDF (pymupdf4llm dependency).
# libgl1 / libglib2.0-0 prevent headless rendering errors on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Install Python Dependencies ----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Copy App ----
COPY . .
RUN chmod +x /app/start.sh

# ---- Expose Port ----
EXPOSE 10000

# ---- Start Command ----
CMD ["/app/start.sh"]
