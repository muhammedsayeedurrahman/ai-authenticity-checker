# ============================================================
# Stage 1: Build React frontend
# ============================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /build

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ .

# Vite bakes these into the bundle at build time
ARG VITE_SUPABASE_URL=""
ARG VITE_SUPABASE_ANON_KEY=""

RUN npm run build

# ============================================================
# Stage 2: Python runtime
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# System dependencies for OpenCV, audio processing, health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY main.py .
COPY api/ api/
COPY core/ core/
COPY core_models/ core_models/
COPY pipeline/ pipeline/
COPY db/ db/
COPY configs/ configs/
COPY alembic/ alembic/
COPY alembic.ini .
COPY assets/ assets/

# Copy built frontend from stage 1
COPY --from=frontend-builder /build/dist/ frontend/dist/

# Copy local model weights (present in build context for HF Spaces,
# volume-mounted in docker-compose — see .dockerignore note)
COPY models/ models/

# HuggingFace cache directory (runtime downloads go here)
ENV HF_HOME=/app/.hf_cache

# Create non-root user and set ownership
RUN groupadd -r proofyx && useradd -r -g proofyx -d /app proofyx \
    && mkdir -p /app/data /app/.hf_cache \
    && chown -R proofyx:proofyx /app

USER proofyx

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7861/api/v1/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7861", "--workers", "1"]
