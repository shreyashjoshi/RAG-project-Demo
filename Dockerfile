# Multi-stage build producing a minimal distroless runtime image for the RAG Flask app
# Build Stage ---------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

# Install build deps (adjust if chromadb / unstructured needs extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-fetch the sentence-transformers model to bake into image (optional, speeds cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application source
COPY . .

# Remove any pyc files
RUN find . -type f -name '*.pyc' -delete

# Runtime Stage (Distroless) ------------------------------------------------
# Using distroless python image (nonroot variant). Includes Python interpreter but no shell.
FROM gcr.io/distroless/python3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    CHROMA_PATH=/app/chroma \
    DATA_PATH=/app/data/books

WORKDIR /app
RUN mkdir -p /app/data/books

# Copy virtual environment and application code from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Expose Flask port
EXPOSE 5000

# Define volumes for persistence (optional)
#VOLUME ["/app/chroma", "/app/data/books"]

# Healthcheck (simple TCP check via python) - optional; distroless has no /bin/sh
# Uncomment if you want container orchestrators to leverage it.
# HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["/opt/venv/bin/python", "-c", "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',5000)); s.close()"]

# Start the application
ENTRYPOINT ["app_1.py"]



