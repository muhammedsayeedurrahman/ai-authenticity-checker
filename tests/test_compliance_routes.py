"""API-level tests for /api/v1/compliance/orgs and org-scoped API keys."""

from __future__ import annotations

import os
from unittest.mock import patch

import jwt
import pytest

os.environ.setdefault("DATABASE_URL", "")

JWT_SECRET = "test-secret-for-compliance-routes"


def _bearer_for(user_id: str, email: str = "user@example.com") -> dict[str, str]:
    token = jwt.encode(
        {"sub": user_id, "email": email, "aud": "authenticated"}, JWT_SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def jwt_auth():
    with patch("core.auth._get_jwt_secret", return_value=JWT_SECRET):
        yield


class TestCreateOrg:
    def test_requires_jwt_not_api_key(self, client):
        resp = client.post(
            "/api/v1/compliance/orgs",
            json={"name": "Acme", "slug": "acme"},
        )
        assert resp.status_code == 401

    def test_jwt_user_can_create_org(self, client, jwt_auth):
        resp = client.post(
            "/api/v1/compliance/orgs",
            json={"name": "Acme Platform", "slug": "acme-co"},
            headers=_bearer_for("user-1"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["slug"] == "acme-co"
        assert body["data"]["role"] == "owner"

    def test_duplicate_slug_rejected(self, client, jwt_auth):
        client.post(
            "/api/v1/compliance/orgs",
            json={"name": "Acme", "slug": "dup-slug"},
            headers=_bearer_for("user-1"),
        )
        resp = client.post(
            "/api/v1/compliance/orgs",
            json={"name": "Other", "slug": "dup-slug"},
            headers=_bearer_for("user-2"),
        )
        assert resp.status_code == 409


class TestListMyOrgs:
    def test_returns_created_org(self, client, jwt_auth):
        client.post(
            "/api/v1/compliance/orgs",
            json={"name": "Acme", "slug": "acme-list"},
            headers=_bearer_for("user-3"),
        )
        resp = client.get("/api/v1/compliance/orgs/me", headers=_bearer_for("user-3"))
        assert resp.status_code == 200
        slugs = [o["slug"] for o in resp.json()["data"]]
        assert "acme-list" in slugs

    def test_empty_for_user_with_no_orgs(self, client, jwt_auth):
        resp = client.get("/api/v1/compliance/orgs/me", headers=_bearer_for("user-lonely"))
        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestApiKeyLifecycle:
    def _create_org(self, client, jwt_auth, user_id, slug):
        resp = client.post(
            "/api/v1/compliance/orgs",
            json={"name": "Acme", "slug": slug},
            headers=_bearer_for(user_id),
        )
        return resp.json()["data"]["id"]

    def test_owner_can_create_and_list_keys(self, client, jwt_auth):
        org_id = self._create_org(client, jwt_auth, "owner-1", "acme-keys-1")

        create_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "prod"},
            headers=_bearer_for("owner-1"),
        )
        assert create_resp.status_code == 200
        body = create_resp.json()
        assert body["raw_key"].startswith("pfx_live_")
        assert body["data"]["label"] == "prod"

        list_resp = client.get(
            f"/api/v1/compliance/orgs/{org_id}/api-keys", headers=_bearer_for("owner-1"),
        )
        assert list_resp.status_code == 200
        keys = list_resp.json()["data"]
        assert len(keys) == 1
        assert "raw_key" not in keys[0]

    def test_multi_org_user_can_manage_second_org_without_org_header(self, client, jwt_auth):
        """A JWT user belonging to two orgs must be able to manage the
        second (non-default) org's API keys purely via the org_id in the
        URL path — get_principal's header-based org resolution (which
        defaults to the user's first-joined org when no
        X-Proofyx-Org-Id is sent) must not be the sole authorization
        signal for these org_id-in-path endpoints."""
        self._create_org(client, jwt_auth, "multi-org-user", "acme-multi-1")
        second_org_id = self._create_org(client, jwt_auth, "multi-org-user", "acme-multi-2")

        # No X-Proofyx-Org-Id header sent — only the Bearer token.
        resp = client.post(
            f"/api/v1/compliance/orgs/{second_org_id}/api-keys",
            json={"label": "second-org-key"},
            headers=_bearer_for("multi-org-user"),
        )
        assert resp.status_code == 200

    def test_non_member_cannot_create_key(self, client, jwt_auth):
        org_id = self._create_org(client, jwt_auth, "owner-2", "acme-keys-2")

        resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "intruder"},
            headers=_bearer_for("stranger"),
        )
        assert resp.status_code == 403

    def test_new_key_authenticates_analyze_endpoint(self, client, jwt_auth, dummy_image_bytes):
        org_id = self._create_org(client, jwt_auth, "owner-3", "acme-keys-3")
        create_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "ci"},
            headers=_bearer_for("owner-3"),
        )
        raw_key = create_resp.json()["raw_key"]

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
                headers={"X-API-Key": raw_key},
            )
        assert resp.status_code == 200

    def test_revoke_key_then_it_no_longer_authenticates(self, client, jwt_auth, dummy_image_bytes):
        org_id = self._create_org(client, jwt_auth, "owner-4", "acme-keys-4")
        create_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "temp"},
            headers=_bearer_for("owner-4"),
        )
        key_data = create_resp.json()
        raw_key = key_data["raw_key"]
        key_id = key_data["data"]["id"]

        revoke_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys/{key_id}/revoke",
            headers=_bearer_for("owner-4"),
        )
        assert revoke_resp.status_code == 200
        assert revoke_resp.json()["revoked"] is True

        # Revoked org key falls through to legacy pool check, which (with
        # no pool configured in this test) rejects it as unconfigured auth.
        with patch("core.auth.get_pool", return_value=None):
            resp = client.post(
                "/api/v1/analyze/image",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                headers={"X-API-Key": raw_key},
            )
        assert resp.status_code == 403
