"""
ProofyX -- FastAPI entry point.

Serves:
    /api/v1 -> REST API Endpoints
    /docs   -> Swagger auto-generated documentation
    /*      -> React SPA (from frontend/dist/)
"""

from __future__ import annotations

import os
import sys
import logging

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Import config first -- triggers load_dotenv() before other modules read env vars
import core.config  # noqa: F401

from api.routes import router as api_router, limiter
from core.secrets import get_active_pools

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("proofyx")

# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and eagerly load ML models at startup."""
    from db.database import init_db, close_db
    from core.pipeline import get_registry

    # Database
    logger.info("Initializing database...")
    await init_db()

    # ML Models
    logger.info("Loading ML models (this may take a moment on first run)...")
    reg = get_registry()
    logger.info(
        "Model loading complete: %d loaded, %d missing",
        len(reg.loaded), len(reg.missing),
    )
    yield

    # Cleanup
    await close_db()


app = FastAPI(
    title="ProofyX API",
    description="AI-powered multimodal deepfake & manipulation detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────
# Middleware: CORS
# ──────────────────────────────────────────────

_cors_raw = os.environ.get("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if not _cors_origins:
    _cors_origins = [
        "http://localhost:7861", "http://127.0.0.1:7861",
        "http://localhost:5173", "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization", "Accept"],
)

# ──────────────────────────────────────────────
# Middleware: Security Headers
# ──────────────────────────────────────────────

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to all responses."""

    async def dispatch(self, request: StarletteRequest, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ──────────────────────────────────────────────
# Middleware: Rate Limiting
# ──────────────────────────────────────────────

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ──────────────────────────────────────────────
# REST API routes
# ──────────────────────────────────────────────

app.include_router(api_router, prefix="/api/v1", tags=["Analysis"])

# Static files (project-level assets: logo, etc.)
assets_dir = os.path.join(ROOT_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/app-assets", StaticFiles(directory=assets_dir), name="app-assets")

# ──────────────────────────────────────────────
# React SPA (production build from frontend/dist/)
# ──────────────────────────────────────────────

frontend_dist = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.isdir(frontend_dist):
    _spa_index = os.path.join(frontend_dist, "index.html")

    @app.get("/")
    async def spa_root():
        """Serve React SPA root."""
        return FileResponse(_spa_index)

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        """Serve React SPA -- try static file first, fall back to index.html."""
        file_path = os.path.join(frontend_dist, path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(_spa_index)
else:
    logger.warning("frontend/dist/ not found — run 'cd frontend && npm run build' to serve the React UI")

# ──────────────────────────────────────────────
# Key Pool Status
# ──────────────────────────────────────────────

active_pools = get_active_pools()
if active_pools:
    logger.info("API key pools configured:")
    for service, pool in active_pools.items():
        logger.info("  %s: %d key(s)", service, pool.size)
    if "PROOFYX_API_KEY" in active_pools:
        logger.info("  API auth: ENABLED (X-API-Key required for /api/v1/analyze/*)")
else:
    logger.info("No API key pools configured -- running in dev mode (unauthenticated)")

logger.info("ProofyX ready:")
logger.info("  UI:        http://127.0.0.1:7861/")
logger.info("  REST API:  http://127.0.0.1:7861/api/v1")
logger.info("  Swagger:   http://127.0.0.1:7861/docs")

# ──────────────────────────────────────────────
# Standalone Run
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=7861,
        reload=False,
        log_level="info",
    )
