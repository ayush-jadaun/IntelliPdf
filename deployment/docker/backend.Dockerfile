# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first
COPY backend/requirements.txt .

# Install Python dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

# Download spaCy model with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m spacy download en_core_web_sm

COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic/alembic.ini .
COPY backend/start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]