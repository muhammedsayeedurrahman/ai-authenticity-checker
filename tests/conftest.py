"""Shared fixtures for ProofyX tests."""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Force SQLite for tests
os.environ.setdefault("DATABASE_URL", "")
# Prevent the compliance SLA background monitor from starting during the
# TestClient lifespan (see main.py::lifespan) — it would otherwise leak a
# task across every API test in the suite.
os.environ.setdefault("PROOFYX_SLA_MONITOR_ENABLED", "0")


@pytest.fixture()
def _mock_registry():
    """Patch the model registry so tests don't load real ML models."""
    import api.routes  # noqa: F401

    mock_reg = MagicMock()
    mock_reg.loaded = ["vit", "efficientnet"]
    mock_reg.missing = ["dino"]
    mock_reg.models = {}
    mock_reg.get_status.return_value = {
        "loaded": ["vit", "efficientnet"],
        "missing": ["dino"],
        "total": 2,
        "corefakenet_ready": False,
    }
    with patch("core.pipeline._registry", mock_reg), \
         patch("core.pipeline.get_registry", return_value=mock_reg), \
         patch("api.routes.get_registry", return_value=mock_reg):
        yield mock_reg


@pytest.fixture()
def client(_mock_registry, tmp_path, monkeypatch):
    """FastAPI TestClient with mocked model registry and an isolated,
    per-test SQLite DB.

    Two bugs this fixes at once:
      1. `TestClient(app)` used bare (no `with`) never runs the ASGI
         lifespan, so `init_db()` never creates the `analyses` table.
         This passed locally only because the dev machine's real
         proofyx_history.db already had that table from actually running
         main.py - a fresh CI checkout has no such file, so every test
         hitting an endpoint that writes history failed with
         "no such table: analyses". `with TestClient(app) as c:` runs
         startup/shutdown for real (registry is still mocked, so no real
         models load).
      2. Without an isolated DB, those same writes were landing in the
         real dev-machine proofyx_history.db - local test runs were
         quietly inserting fake "test.jpg" rows into real analysis
         history. Point DATABASE_URL at a pytest tmp_path file instead,
         and reset the cached engine/session factory so the new URL
         actually takes effect (db.database caches them as module
         globals).
    """
    db_path = tmp_path / "test_history.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    import db.database as database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_session_factory", None)

    from main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def dummy_image_bytes() -> bytes:
    """Generate a minimal valid JPEG in memory."""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def oversized_bytes() -> bytes:
    """Generate bytes exceeding the 50MB upload limit."""
    return b"\x00" * (51 * 1024 * 1024)
