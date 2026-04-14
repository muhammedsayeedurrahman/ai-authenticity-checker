"""
ProofyX — FastAPI + Gradio entry point.

Serves:
    /ui     → Gradio Dashboard (cybersecurity-themed UI)
    /api/v1 → REST API Endpoints
    /docs   → Swagger auto-generated documentation
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
from fastapi.staticfiles import StaticFiles
import gradio as gr

# Import config first — triggers load_dotenv() before other modules read env vars
import core.config  # noqa: F401

from api.routes import router as api_router
from core.secrets import get_active_pools
from ui.gradio_app import create_gradio_app
from ui.theme import CUSTOM_CSS, FORCE_DARK_JS, create_theme

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

app = FastAPI(
    title="ProofyX API",
    description="AI-powered multimodal deepfake & manipulation detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# REST API routes
app.include_router(api_router, prefix="/api/v1", tags=["Analysis"])

# Static files (logo, assets)
assets_dir = os.path.join(ROOT_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# ──────────────────────────────────────────────
# Gradio Mount
# ──────────────────────────────────────────────

logger.info("Creating Gradio app...")
gradio_app = create_gradio_app()

app = gr.mount_gradio_app(
    app,
    gradio_app,
    path="/ui",
    allowed_paths=[assets_dir],
    theme=create_theme(),
    css=CUSTOM_CSS,
    js=FORCE_DARK_JS,
)

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
    logger.info("No API key pools configured — running in dev mode (unauthenticated)")

logger.info("ProofyX ready:")
logger.info("  Dashboard: http://127.0.0.1:7861/ui")
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
