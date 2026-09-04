"""API-level tests for /api/v1/compliance/orgs/{org_id}/webhooks."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest

os.environ.setdefault("DATABASE_URL", "")

JWT_SECRET = "test-secret-for-webhook-routes"


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


class TestRegisterWebhook:
    def test_register_and_list(self, client, jwt_auth):
        org_id = _make_org(client, "owner-1", "acme-wh-1")
        resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks",
            json={"url": "https://example.com/hook", "event_types": ["sla.breached"]},
            headers=_bearer_for("owner-1"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["secret"]
        assert body["data"]["url"] == "https://example.com/hook"

        list_resp = client.get(
            f"/api/v1/compliance/orgs/{org_id}/webhooks", headers=_bearer_for("owner-1"),
        )
        endpoints = list_resp.json()["data"]
        assert len(endpoints) == 1
        assert "secret" not in endpoints[0]

    def test_rejects_non_https_url(self, client, jwt_auth):
        org_id = _make_org(client, "owner-2", "acme-wh-2")
        resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks",
            json={"url": "http://example.com/hook"},
            headers=_bearer_for("owner-2"),
        )
        assert resp.status_code == 400

    def test_rejects_private_ip_url(self, client, jwt_auth):
        org_id = _make_org(client, "owner-3", "acme-wh-3")
        resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks",
            json={"url": "https://127.0.0.1/hook"},
            headers=_bearer_for("owner-3"),
        )
        assert resp.status_code == 400

    def test_non_member_cannot_register(self, client, jwt_auth):
        org_id = _make_org(client, "owner-4", "acme-wh-4")
        resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks",
            json={"url": "https://example.com/hook"},
            headers=_bearer_for("stranger"),
        )
        assert resp.status_code == 403


class TestRevokeAndTestWebhook:
    def test_revoke(self, client, jwt_auth):
        org_id = _make_org(client, "owner-5", "acme-wh-5")
        create_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks",
            json={"url": "https://example.com/hook"},
            headers=_bearer_for("owner-5"),
        )
        endpoint_id = create_resp.json()["data"]["id"]

        revoke_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks/{endpoint_id}/revoke",
            headers=_bearer_for("owner-5"),
        )
        assert revoke_resp.status_code == 200
        assert revoke_resp.json()["revoked"] is True

    def test_test_delivery_reports_success(self, client, jwt_auth):
        org_id = _make_org(client, "owner-6", "acme-wh-6")
        create_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks",
            json={"url": "https://example.com/hook"},
            headers=_bearer_for("owner-6"),
        )
        endpoint_id = create_resp.json()["data"]["id"]

        response = MagicMock(status_code=200)

        async def aiter_bytes():
            yield b"ok"
        response.aiter_bytes = aiter_bytes

        class _StreamCM:
            async def __aenter__(self):
                return response

            async def __aexit__(self, *exc):
                return False

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=_StreamCM())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks._resolve_public_addresses", return_value=["93.184.216.34"]), \
             patch("httpx.AsyncClient", return_value=mock_client):
            resp = client.post(
                f"/api/v1/compliance/orgs/{org_id}/webhooks/{endpoint_id}/test",
                headers=_bearer_for("owner-6"),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["delivered"] is True
        assert body["status_code"] == 200

    def test_test_delivery_on_unknown_endpoint_404s(self, client, jwt_auth):
        org_id = _make_org(client, "owner-7", "acme-wh-7")
        resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks/does-not-exist/test",
            headers=_bearer_for("owner-7"),
        )
        assert resp.status_code == 404


class TestWebhookNotifiedOnIngest:
    def test_flagged_ingest_enqueues_a_delivery(self, client, jwt_auth, dummy_image_bytes):
        from tests.test_compliance_ingest import FLAGGED_RESULT

        org_id = _make_org(client, "owner-8", "acme-wh-8")
        client.post(
            f"/api/v1/compliance/orgs/{org_id}/webhooks",
            json={"url": "https://example.com/hook"},
            headers=_bearer_for("owner-8"),
        )
        key_resp = client.post(
            f"/api/v1/compliance/orgs/{org_id}/api-keys",
            json={"label": "ci"}, headers=_bearer_for("owner-8"),
        )
        raw_key = key_resp.json()["raw_key"]

        with patch("api.compliance_routes.analyze_image", return_value=FLAGGED_RESULT):
            client.post(
                "/api/v1/compliance/content",
                files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
                data={"external_content_ref": "post-wh", "media_type": "image"},
                headers={"X-API-Key": raw_key},
            )

        deliveries_resp = client.get(
            f"/api/v1/compliance/orgs/{org_id}/webhooks/deliveries", headers=_bearer_for("owner-8"),
        )
        deliveries = deliveries_resp.json()["data"]
        event_types = {d["event_type"] for d in deliveries}
        assert "content.labeled" in event_types
        assert "sla.started" in event_types
        assert all(d["status"] == "pending" for d in deliveries)
