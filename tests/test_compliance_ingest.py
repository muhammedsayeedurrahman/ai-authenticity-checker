"""API-level tests for the compliance ingestion, SLA, and audit-log endpoints."""

from __future__ import annotations

import os
from unittest.mock import patch

import jwt
import pytest

os.environ.setdefault("DATABASE_URL", "")

JWT_SECRET = "test-secret-for-compliance-ingest"


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


def _make_org(client, user_id: str, slug: str) -> str:
    resp = client.post(
        "/api/v1/compliance/orgs",
        json={"name": "Acme", "slug": slug},
        headers=_bearer_for(user_id),
    )
    assert resp.status_code == 200
    return resp.json()["data"]["id"]


FLAGGED_RESULT = {
    "risk_score": 0.92, "risk_percent": 92.0, "verdict": "AI-GENERATED",
    "confidence": "HIGH", "media_type": "image",
    "cybercrime_risk": {
        "category": "synthetic_identity", "label": "Synthetic identity / identity-fraud pattern",
        "description": "", "advisory": "", "signals": [], "disclaimer": "x",
    },
    "compliance_label": {
        "label_code": "synthetically_generated",
        "label_display": "Synthetically generated (AI content)",
        "requires_visible_label": True, "requires_embedded_metadata": True,
        "label_basis": ["Face manipulation detected"],
        "regulatory_basis": "India IT Rules 2026", "ruleset_version": "in-it-rules-2026.v1",
        "detector_version": "proofyx/learned", "risk_score": 0.92, "confidence": "HIGH",
        "recommended_action": "label_and_review", "sla_applies": True,
        "sla_deadline_seconds": 10800, "assessed_at": "2026-09-04T00:00:00+00:00",
        "disclaimer": "not legal advice",
    },
}

CLEAN_RESULT = {
    "risk_score": 0.05, "risk_percent": 5.0, "verdict": "LIKELY AUTHENTIC",
    "confidence": "HIGH", "media_type": "image",
    "cybercrime_risk": {"category": "none", "label": "", "description": "", "advisory": "", "signals": [], "disclaimer": ""},
    "compliance_label": {
        "label_code": "no_synthetic_indicators", "label_display": "No synthetic-content indicators",
        "requires_visible_label": False, "requires_embedded_metadata": False,
        "label_basis": [], "regulatory_basis": "India IT Rules 2026",
        "ruleset_version": "in-it-rules-2026.v1", "detector_version": "proofyx/learned",
        "risk_score": 0.05, "confidence": "HIGH", "recommended_action": "none",
        "sla_applies": False, "sla_deadline_seconds": None,
        "assessed_at": "2026-09-04T00:00:00+00:00", "disclaimer": "not legal advice",
    },
}


class TestIngestContent:
    def test_flagged_content_opens_sla_clock_and_audit_trail(
        self, client, jwt_auth, dummy_image_bytes,
    ):
        org_id = _make_org(client, "owner-1", "acme-ingest-1")
        key_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "ci"}, headers=_bearer_for("owner-1"),
        )
        raw_key = key_resp.json()["raw_key"]

        with patch("api.compliance_routes.analyze_image", return_value=FLAGGED_RESULT):
            resp = client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={"external_content_ref": "post-999", "media_type": "image"},
                headers={"X-API-Key": raw_key},
            )
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["label"]["label_code"] == "synthetically_generated"
        assert body["sla"] is not None
        assert body["sla"]["status"] == "running"
        assert body["sla"]["effective_status"] == "running"

        audit_resp = client.get(
            "/api/v1/compliance/audit-log", headers={"X-API-Key": raw_key},
        )
        assert audit_resp.status_code == 200
        audit_body = audit_resp.json()
        assert audit_body["chain_verified"] is True
        event_types = [e["event_type"] for e in audit_body["data"]]
        assert "content.analyzed" in event_types
        assert "content.labeled" in event_types
        assert "sla.started" in event_types

    def test_clean_content_does_not_open_sla_clock(self, client, jwt_auth, dummy_image_bytes):
        org_id = _make_org(client, "owner-2", "acme-ingest-2")
        key_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "ci"}, headers=_bearer_for("owner-2"),
        )
        raw_key = key_resp.json()["raw_key"]

        with patch("api.compliance_routes.analyze_image", return_value=CLEAN_RESULT):
            resp = client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={"external_content_ref": "post-clean", "media_type": "image"},
                headers={"X-API-Key": raw_key},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["sla"] is None

    def test_legacy_api_key_without_org_gets_403(self, client, dummy_image_bytes):
        from core.secrets import KeyPool

        pool = KeyPool("PROOFYX_API_KEY", ["legacy-key"])
        with patch("core.auth.get_pool", return_value=pool):
            resp = client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={"external_content_ref": "post-x", "media_type": "image"},
                headers={"X-API-Key": "legacy-key"},
            )
        assert resp.status_code == 403

    def test_flagged_by_complaint_overrides_sla_applies(
        self, client, jwt_auth, dummy_image_bytes,
    ):
        """Even a result whose cybercrime category is 'none' should open an
        SLA clock when the caller explicitly reports a complaint was filed."""
        org_id = _make_org(client, "owner-3", "acme-ingest-3")
        key_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "ci"}, headers=_bearer_for("owner-3"),
        )
        raw_key = key_resp.json()["raw_key"]

        flagged_but_no_category = {
            **FLAGGED_RESULT,
            "cybercrime_risk": {"category": "none", "label": "", "description": "", "advisory": "", "signals": [], "disclaimer": ""},
            "compliance_label": {**FLAGGED_RESULT["compliance_label"], "sla_applies": False, "sla_deadline_seconds": None},
        }
        with patch("api.compliance_routes.analyze_image", return_value=flagged_but_no_category):
            resp = client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={
                    "external_content_ref": "post-complaint", "media_type": "image",
                    "flagged_by_complaint": "true",
                },
                headers={"X-API-Key": raw_key},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["sla"] is not None


class TestContentAction:
    def test_action_closes_clock_and_records_audit_entry(
        self, client, jwt_auth, dummy_image_bytes,
    ):
        org_id = _make_org(client, "owner-4", "acme-ingest-4")
        key_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "ci"}, headers=_bearer_for("owner-4"),
        )
        raw_key = key_resp.json()["raw_key"]

        with patch("api.compliance_routes.analyze_image", return_value=FLAGGED_RESULT):
            ingest_resp = client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={"external_content_ref": "post-action", "media_type": "image"},
                headers={"X-API-Key": raw_key},
            )
        label_id = ingest_resp.json()["data"]["content_label_id"]

        action_resp = client.post(
            f"/api/v1/compliance/content/{label_id}/action",
            json={"action": "removed", "notes": "taken down per policy"},
            headers={"X-API-Key": raw_key},
        )
        assert action_resp.status_code == 200
        body = action_resp.json()["data"]
        assert body["status"] == "met"
        assert body["action"] == "removed"

        sla_list = client.get("/api/v1/compliance/sla", headers={"X-API-Key": raw_key})
        assert sla_list.json()["data"][0]["status"] == "met"

    def test_duplicate_action_call_returns_409_and_does_not_flip_status(
        self, client, jwt_auth, dummy_image_bytes,
    ):
        org_id = _make_org(client, "owner-6", "acme-ingest-6")
        key_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "ci"}, headers=_bearer_for("owner-6"),
        )
        raw_key = key_resp.json()["raw_key"]

        with patch("api.compliance_routes.analyze_image", return_value=FLAGGED_RESULT):
            ingest_resp = client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={"external_content_ref": "post-dup-action", "media_type": "image"},
                headers={"X-API-Key": raw_key},
            )
        label_id = ingest_resp.json()["data"]["content_label_id"]

        first = client.post(
            f"/api/v1/compliance/content/{label_id}/action",
            json={"action": "removed"},
            headers={"X-API-Key": raw_key},
        )
        assert first.status_code == 200
        assert first.json()["data"]["status"] == "met"

        second = client.post(
            f"/api/v1/compliance/content/{label_id}/action",
            json={"action": "removed"},
            headers={"X-API-Key": raw_key},
        )
        assert second.status_code == 409

        sla_list = client.get("/api/v1/compliance/sla", headers={"X-API-Key": raw_key})
        assert sla_list.json()["data"][0]["status"] == "met"

    def test_action_on_content_with_no_clock_returns_404(
        self, client, jwt_auth, dummy_image_bytes,
    ):
        org_id = _make_org(client, "owner-5", "acme-ingest-5")
        key_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "ci"}, headers=_bearer_for("owner-5"),
        )
        raw_key = key_resp.json()["raw_key"]

        with patch("api.compliance_routes.analyze_image", return_value=CLEAN_RESULT):
            ingest_resp = client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={"external_content_ref": "post-no-clock", "media_type": "image"},
                headers={"X-API-Key": raw_key},
            )
        label_id = ingest_resp.json()["data"]["content_label_id"]

        resp = client.post(
            f"/api/v1/compliance/content/{label_id}/action",
            json={"action": "cleared_false_positive"},
            headers={"X-API-Key": raw_key},
        )
        assert resp.status_code == 404


class TestCrossTenantIsolation:
    def test_org_a_key_cannot_see_org_b_audit_log(self, client, jwt_auth, dummy_image_bytes):
        org_a = _make_org(client, "owner-a", "acme-tenant-a")
        org_b = _make_org(client, "owner-b", "acme-tenant-b")
        key_a = client.post(
            f"/api/v1/compliance/orgs/{org_a}/api-keys",
            json={"label": "a"}, headers=_bearer_for("owner-a"),
        ).json()["raw_key"]
        key_b = client.post(
            f"/api/v1/compliance/orgs/{org_b}/api-keys",
            json={"label": "b"}, headers=_bearer_for("owner-b"),
        ).json()["raw_key"]

        with patch("api.compliance_routes.analyze_image", return_value=FLAGGED_RESULT):
            client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={"external_content_ref": "b-content", "media_type": "image"},
                headers={"X-API-Key": key_b},
            )

        audit_a = client.get("/api/v1/compliance/audit-log", headers={"X-API-Key": key_a})
        assert audit_a.json()["data"] == []

        sla_a = client.get("/api/v1/compliance/sla", headers={"X-API-Key": key_a})
        assert sla_a.json()["data"] == []
