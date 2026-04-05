#!/bin/sh
set -eu

PORT_VALUE="${PORT:-10000}"
# Use 2 workers in production so uploads don't block health checks.
# Keep it at 2 (not auto) because each worker holds DB connection pool + httpx client.
WORKERS="${WEB_CONCURRENCY:-2}"

echo "Applying database migrations..."
python -m alembic upgrade head

echo "Starting API server on port ${PORT_VALUE} with ${WORKERS} worker(s)..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT_VALUE}" \
    --workers "${WORKERS}" \
    --timeout-keep-alive 75
