#!/bin/sh
set -eu

PORT_VALUE="${PORT:-10000}"

echo "Applying database migrations..."
python -m alembic upgrade head

echo "Starting API server on port ${PORT_VALUE}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT_VALUE}"
