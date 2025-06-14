#!/bin/bash
set -e

# Run Alembic migrations
alembic upgrade head

# Start the app server
exec uvicorn app.main:app --host 0.0.0.0 --port 8000