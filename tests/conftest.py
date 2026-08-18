"""Shared fixtures for ProofyX tests."""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Force SQLite for tests
os.environ.setdefault("DATABASE_URL", "")


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
def client(_mock_registry, monkeypatch):
    """FastAPI TestClient with mocked model registry and an isolated in-memory DB.

    Entering TestClient as a context manager runs the app's `lifespan`, which
    calls `init_db()` to create tables. The engine/session factory are
    monkeypatched beforehand so that call creates tables on an isolated
    in-memory SQLite DB instead of the real on-disk one.
    """
    import db.database as database

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(database, "_session_factory", factory)

    from main import app
    with TestClient(app) as test_client:
        yield test_client


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
