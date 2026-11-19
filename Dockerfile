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

# ---- Run as non-root ----
# The app only reads /app (bytecode writes are disabled) and writes to /tmp
# and stdout, so a locked-down user is safe and limits container escapes.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

# ---- Expose Port ----
EXPOSE 10000

# ---- Healthcheck ----
# ${PORT} is expanded at runtime by the shell form; start.sh defaults to 10000.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-10000}/health" || exit 1

# ---- Start Command ----
CMD ["/app/start.sh"]
