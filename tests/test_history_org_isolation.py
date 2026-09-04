"""Regression tests for the CRITICAL cross-tenant /history leak.

An org-scoped API key must never see another org's (or another
credential's) analysis history through GET /history or
GET /history/{id} — previously both endpoints depended on
get_current_user, which collapses every API-key principal to None,
causing db/history.py to skip its scoping filter entirely.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import jwt
import pytest

os.environ.setdefault("DATABASE_URL", "")

JWT_SECRET = "test-secret-for-history-isolation"


def _bearer_for(user_id: str) -> dict[str, str]:
    token = jwt.encode(
        {"sub": user_id, "email": f"{user_id}@example.com", "aud": "authenticated"},
        JWT_SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def jwt_auth():
    with patch("core.auth._get_jwt_secret", return_value=JWT_SECRET):
        yield


def _make_org_and_key(client, user_id: str, slug: str) -> tuple[str, str]:
    org_resp = client.post(
        "/api/v1/compliance/orgs", json={"name": "Acme", "slug": slug}, headers=_bearer_for(user_id),
    )
    org_id = org_resp.json()["data"]["id"]
    key_resp = client.post(
        f"/api/v1/compliance/orgs/{org_id}/api-keys", json={"label": "k"}, headers=_bearer_for(user_id),
    )
    return org_id, key_resp.json()["raw_key"]


MOCK_RESULT = {
    "risk_score": 0.9, "risk_percent": 90.0, "verdict": "AI-GENERATED", "confidence": "HIGH",
    "risk_level": "HIGH", "model_scores": {}, "fusion_mode": "weighted_avg",
    "face_detected": False, "face_aligned": False, "gradcam_image": None, "original_image": None,
    "models_used": 1, "processing_time_ms": 10.0, "explanation": "", "media_type": "image",
    "model_agreement": "",
}


class TestHistoryOrgIsolation:
    def test_org_a_key_cannot_see_org_b_history(self, client, jwt_auth, dummy_image_bytes):
        org_a, key_a = _make_org_and_key(client, "owner-a", "hist-org-a")
        org_b, key_b = _make_org_and_key(client, "owner-b", "hist-org-b")

        with patch("api.routes.analyze_image", return_value=MOCK_RESULT):
            client.post(
                "/api/v1/analyze/image",
                files={"file": ("secret.jpg", dummy_image_bytes, "image/jpeg")},
                headers={"X-API-Key": key_b},
            )

        # Org A's own key must see zero history — not org B's analysis.
        resp = client.get("/api/v1/history", headers={"X-API-Key": key_a})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []
        assert body["total"] == 0

    def test_org_b_key_sees_its_own_history(self, client, jwt_auth, dummy_image_bytes):
        org_a, key_a = _make_org_and_key(client, "owner-c", "hist-org-c")
        org_b, key_b = _make_org_and_key(client, "owner-d", "hist-org-d")

        with patch("api.routes.analyze_image", return_value=MOCK_RESULT):
            client.post(
                "/api/v1/analyze/image",
                files={"file": ("mine.jpg", dummy_image_bytes, "image/jpeg")},
                headers={"X-API-Key": key_b},
            )

        resp = client.get("/api/v1/history", headers={"X-API-Key": key_b})
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_get_specific_analysis_by_id_is_org_scoped(self, client, jwt_auth, dummy_image_bytes):
        org_a, key_a = _make_org_and_key(client, "owner-e", "hist-org-e")
        org_b, key_b = _make_org_and_key(client, "owner-f", "hist-org-f")

        with patch("api.routes.analyze_image", return_value=MOCK_RESULT):
            analyze_resp = client.post(
                "/api/v1/analyze/image",
                files={"file": ("secret.jpg", dummy_image_bytes, "image/jpeg")},
                headers={"X-API-Key": key_b},
            )
        analysis_id = analyze_resp.json()["data"]["id"]

        # Org A guessing/enumerating org B's analysis id must get 404, not the data.
        resp = client.get(f"/api/v1/history/{analysis_id}", headers={"X-API-Key": key_a})
        assert resp.status_code == 404

        # Org B itself can still retrieve it.
        own_resp = client.get(f"/api/v1/history/{analysis_id}", headers={"X-API-Key": key_b})
        assert own_resp.status_code == 200
