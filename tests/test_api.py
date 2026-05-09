"""Integration tests for ProofyX REST API endpoints."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from PIL import Image


# ──────────────────────────────────────────────
# Health & System Endpoints
# ──────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "models_loaded" in data

    def test_health_reports_healthy_when_models_loaded(self, client, _mock_registry):
        _mock_registry.loaded = ["vit"]
        resp = client.get("/api/v1/health")
        assert resp.json()["status"] == "healthy"

    def test_health_reports_degraded_when_no_models(self, client, _mock_registry):
        _mock_registry.loaded = []
        _mock_registry.get_status.return_value = {
            "loaded": [], "missing": ["all"], "total": 0, "corefakenet_ready": False,
        }
        resp = client.get("/api/v1/health")
        assert resp.json()["status"] == "degraded"


class TestModelsStatus:
    def test_models_status_returns_200(self, client):
        resp = client.get("/api/v1/models/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data
        assert "missing" in data
        assert "total" in data
        assert "corefakenet_ready" in data


# ──────────────────────────────────────────────
# Image Analysis — Validation
# ──────────────────────────────────────────────

class TestImageAnalysisValidation:
    def test_rejects_missing_file(self, client):
        resp = client.post("/api/v1/analyze/image")
        assert resp.status_code == 422

    def test_rejects_invalid_image(self, client):
        resp = client.post(
            "/api/v1/analyze/image",
            files={"file": ("test.jpg", b"not-an-image", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_rejects_unsupported_extension(self, client):
        resp = client.post(
            "/api/v1/analyze/image",
            files={"file": ("test.exe", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_rejects_oversized_file(self, client, oversized_bytes):
        resp = client.post(
            "/api/v1/analyze/image",
            files={"file": ("big.jpg", oversized_bytes, "image/jpeg")},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"]

    def test_rejects_invalid_mode(self, client, dummy_image_bytes):
        resp = client.post(
            "/api/v1/analyze/image",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
            params={"mode": "invalid_mode"},
        )
        assert resp.status_code == 422


class TestImageAnalysisSuccess:
    def test_analyze_image_returns_result(self, client, dummy_image_bytes):
        mock_result = {
            "risk_score": 0.42,
            "risk_percent": 42.0,
            "verdict": "UNCERTAIN",
            "confidence": "MEDIUM",
            "risk_level": "MEDIUM",
            "model_agreement": "2/3 models detect manipulation",
            "model_scores": {"vit": 0.6, "efficientnet": 0.3},
            "fusion_mode": "weighted_avg",
            "face_detected": False,
            "face_aligned": False,
            "gradcam_image": None,
            "original_image": None,
            "models_used": 2,
            "processing_time_ms": 150.0,
            "explanation": "Test explanation",
            "media_type": "image",
        }

        with patch("api.routes.analyze_image", return_value=mock_result):
            resp = client.post(
                "/api/v1/analyze/image",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["risk_score"] == 0.42
        assert body["data"]["verdict"] == "UNCERTAIN"


# ──────────────────────────────────────────────
# Video Analysis — Validation
# ──────────────────────────────────────────────

class TestVideoAnalysisValidation:
    def test_rejects_unsupported_video_type(self, client):
        resp = client.post(
            "/api/v1/analyze/video",
            files={"file": ("test.txt", b"data", "text/plain")},
        )
        assert resp.status_code == 400

    def test_rejects_missing_file(self, client):
        resp = client.post("/api/v1/analyze/video")
        assert resp.status_code == 422


# ──────────────────────────────────────────────
# Audio Analysis — Validation
# ──────────────────────────────────────────────

class TestAudioAnalysisValidation:
    def test_rejects_unsupported_audio_type(self, client):
        resp = client.post(
            "/api/v1/analyze/audio",
            files={"file": ("test.pdf", b"data", "application/pdf")},
        )
        assert resp.status_code == 400


# ──────────────────────────────────────────────
# History Endpoints
# ──────────────────────────────────────────────

class TestHistoryEndpoints:
    def test_list_history_returns_200(self, client):
        resp = client.get("/api/v1/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["data"], list)

    def test_list_history_with_limit(self, client):
        resp = client.get("/api/v1/history?limit=5")
        assert resp.status_code == 200

    def test_get_nonexistent_analysis(self, client):
        resp = client.get("/api/v1/history/nonexistent_id")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# API Key Authentication
# ──────────────────────────────────────────────

class TestAPIKeyAuth:
    def test_dev_mode_allows_unauthenticated(self, client, dummy_image_bytes):
        """When no API keys are configured, requests pass through."""
        mock_result = {
            "risk_score": 0.1, "risk_percent": 10.0,
            "verdict": "LIKELY AUTHENTIC", "confidence": "HIGH",
            "risk_level": "LOW", "model_scores": {},
            "fusion_mode": "weighted_avg", "face_detected": False,
            "face_aligned": False, "gradcam_image": None,
            "original_image": None, "models_used": 1,
            "processing_time_ms": 50.0, "explanation": "",
            "media_type": "image", "model_agreement": "",
        }
        with patch("api.routes.analyze_image", return_value=mock_result):
            resp = client.post(
                "/api/v1/analyze/image",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
            )
        assert resp.status_code == 200

    def test_rejects_missing_key_when_configured(self, client, dummy_image_bytes):
        """When API keys are configured, missing header returns 401."""
        from core.secrets import KeyPool
        mock_pool = KeyPool("PROOFYX_API_KEY", ["test-key-123"])
        with patch("api.routes.get_pool", return_value=mock_pool):
            resp = client.post(
                "/api/v1/analyze/image",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
            )
        assert resp.status_code == 401

    def test_rejects_invalid_key(self, client, dummy_image_bytes):
        """When API keys are configured, wrong key returns 403."""
        from core.secrets import KeyPool
        mock_pool = KeyPool("PROOFYX_API_KEY", ["correct-key"])
        with patch("api.routes.get_pool", return_value=mock_pool):
            resp = client.post(
                "/api/v1/analyze/image",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 403
