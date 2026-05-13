# ProofyX Deployment Guide

## System Requirements

### Minimum Viable Deployment (Single-GPU)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | NVIDIA T4 (16GB VRAM) | NVIDIA A10G (24GB VRAM) |
| CPU | 4 cores | 8 cores |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB (models + app) | 50 GB |
| OS | Ubuntu 22.04+ / Windows 11 | Ubuntu 22.04 LTS |

### Memory Estimates

- **ML Models (GPU VRAM):** ~6-8 GB with all models loaded
  - ViT (HuggingFace): ~1.2 GB
  - CLIP ViT-L/14: ~1.5 GB
  - Wav2Vec2-XLSR: ~1.2 GB
  - Local models (EfficientNet, DINOv2, Face, Frequency, Fusion, CorefakeNet): ~2-3 GB total
- **System RAM:** ~4-6 GB for Python runtime, FastAPI workers, and image processing
- **PostgreSQL:** ~256 MB base + data growth

### CPU-Only Mode

ProofyX works without a GPU (auto-detects CPU). Expect 5-10x slower inference.
Set `PROOFYX_TIMEOUT_IMAGE=120` and `PROOFYX_TIMEOUT_VIDEO=1800` for CPU mode.

## Deployment Options

### Docker Compose (Recommended)

```bash
# 1. Copy environment config
cp .env.example .env
# Edit .env with your actual keys

# 2. Download models
python scripts/download_models.py

# 3. Build and run
docker compose up -d

# 4. Run database migrations
docker compose exec proofyx alembic upgrade head
```

### Manual Deployment

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Set up PostgreSQL (or use SQLite for dev)
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/proofyx

# 3. Run migrations
alembic upgrade head

# 4. Download models
python scripts/download_models.py

# 5. Build frontend
cd frontend && npm install && npm run build && cd ..

# 6. Start server
uvicorn main:app --host 0.0.0.0 --port 7861
```

## Database

### PostgreSQL (Production)

Set `DATABASE_URL` in `.env`:
```
DATABASE_URL=postgresql+asyncpg://proofyx:password@localhost:5432/proofyx
```

### SQLite (Development)

When `DATABASE_URL` is not set, ProofyX falls back to `proofyx_history.db` in the project root.

### Migrations

```bash
# Apply all migrations
alembic upgrade head

# Create a new migration after model changes
alembic revision --autogenerate -m "description"

# Rollback one migration
alembic downgrade -1
```

## Authentication

### Supabase Auth (Recommended)

1. Create a Supabase project at https://supabase.com
2. Enable Email/Password auth in Dashboard > Authentication > Providers
3. Set environment variables:

```env
# Backend
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_JWT_SECRET=your-jwt-secret

# Frontend
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...
```

### API Key Auth (Programmatic Access)

For headless/API access without Supabase:
```env
PROOFYX_API_KEY_1=your-secret-key
```

### Dev Mode (No Auth)

When neither Supabase nor API keys are configured, all endpoints are accessible without authentication.

## Billing Integration Points

Billing is not implemented in code (requires a payment provider account).
The following integration points are identified for future implementation:

1. **Usage metering:** `api/routes.py` analysis endpoints - count per-user analyses
2. **Tier enforcement:** `core/auth.py:get_current_user` - check subscription tier from Supabase user metadata
3. **Rate limiting:** `api/routes.py` limiter - adjust limits per subscription tier
4. **Webhook endpoint:** Add `/api/v1/billing/webhook` for Stripe event handling
5. **Frontend:** Add billing page at `/billing` with plan selection and usage dashboard

Recommended payment provider: **Stripe** with Supabase integration via `supabase-stripe` extension.

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | PostgreSQL connection string (falls back to SQLite) |
| `SUPABASE_URL` | No | Supabase project URL |
| `SUPABASE_ANON_KEY` | No | Supabase anonymous key |
| `SUPABASE_JWT_SECRET` | No | Supabase JWT secret for token validation |
| `PROOFYX_API_KEY_1` | No | API key for programmatic access |
| `HF_TOKEN_1` | No | HuggingFace token for gated model downloads |
| `PROOFYX_MAX_CONCURRENT` | No | Max concurrent GPU inferences (default: 1) |
| `PROOFYX_TIMEOUT_IMAGE` | No | Image analysis timeout in seconds (default: 60) |
| `PROOFYX_TIMEOUT_VIDEO` | No | Video analysis timeout in seconds (default: 600) |
| `CORS_ORIGINS` | No | Comma-separated allowed origins |
| `VITE_SUPABASE_URL` | No | Frontend Supabase URL |
| `VITE_SUPABASE_ANON_KEY` | No | Frontend Supabase anon key |
