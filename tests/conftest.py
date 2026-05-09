"""Shared fixtures for ProofyX tests."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def _mock_registry():
    """Patch the model registry so tests don't load real ML models."""
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
def client(_mock_registry):
    """FastAPI TestClient with mocked model registry."""
    # Import here so the mock is active when the app module loads routes
    from main import app
    return TestClient(app)


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
