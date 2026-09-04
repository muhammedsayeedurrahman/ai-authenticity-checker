"""Tests for db/webhook_repo.py — endpoint registration and delivery queue."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "")

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def db_session(monkeypatch):
    import db.compliance_models  # noqa: F401
    import db.database as database
    from db.models import Base

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(database, "_session_factory", factory)

    yield

    await engine.dispose()


class TestEndpointLifecycle:
    async def test_create_returns_secret_once(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        assert endpoint["secret"]

        listed = await repo.list_endpoints("org-1")
        assert "secret" not in listed[0]
        assert "secret_enc" not in listed[0]

    async def test_get_endpoint_secret_decrypts(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        secret = await repo.get_endpoint_secret(endpoint["id"])
        assert secret == endpoint["secret"]

    async def test_list_active_endpoints_for_event_respects_subscription(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        specific = await repo.create_endpoint(
            org_id="org-1", url="https://example.com/a", event_types=["sla.breached"],
        )
        all_events = await repo.create_endpoint(org_id="org-1", url="https://example.com/b")

        subscribed = await repo.list_active_endpoints_for_event("org-1", "sla.breached")
        ids = {e["id"] for e in subscribed}
        assert specific["id"] in ids
        assert all_events["id"] in ids

        not_subscribed = await repo.list_active_endpoints_for_event("org-1", "content.labeled")
        ids2 = {e["id"] for e in not_subscribed}
        assert specific["id"] not in ids2
        assert all_events["id"] in ids2

    async def test_revoke_endpoint(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        assert await repo.revoke_endpoint(endpoint["id"], "org-1") is True

        subscribed = await repo.list_active_endpoints_for_event("org-1", "any.event")
        assert subscribed == []

    async def test_record_delivery_result_resets_on_success(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        await repo.record_delivery_result(endpoint["id"], ok=False)
        await repo.record_delivery_result(endpoint["id"], ok=False)
        mid = await repo.get_endpoint(endpoint["id"])
        assert mid["consecutive_failures"] == 2

        await repo.record_delivery_result(endpoint["id"], ok=True)
        after = await repo.get_endpoint(endpoint["id"])
        assert after["consecutive_failures"] == 0
        assert after["last_success_at"] is not None

    async def test_auto_deactivates_after_threshold(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        for _ in range(3):
            await repo.record_delivery_result(endpoint["id"], ok=False, deactivate_after=3)

        after = await repo.get_endpoint(endpoint["id"])
        assert after["is_active"] is False


class TestNotify:
    async def test_enqueues_for_subscribed_endpoints_only(self):
        from core.webhooks import notify
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        await repo.create_endpoint(org_id="org-1", url="https://example.com/a", event_types=["sla.breached"])
        await repo.create_endpoint(org_id="org-1", url="https://example.com/b", event_types=["content.labeled"])

        count = await notify("org-1", "sla.breached", {"x": 1})
        assert count == 1

    async def test_returns_zero_for_org_with_no_endpoints(self):
        from core.webhooks import notify

        count = await notify("org-with-none", "content.labeled", {})
        assert count == 0


class TestDeliveryQueue:
    async def test_enqueue_and_list_due(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        delivery = await repo.enqueue_delivery(
            org_id="org-1", endpoint_id=endpoint["id"], event_type="content.labeled",
            payload={"x": 1},
        )
        due = await repo.list_due_deliveries()
        assert any(d["id"] == delivery["id"] for d in due)

    async def test_mark_delivered_removes_from_due_list(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        delivery = await repo.enqueue_delivery(
            org_id="org-1", endpoint_id=endpoint["id"], event_type="x", payload={},
        )
        await repo.mark_delivered(delivery["id"], 200)

        due = await repo.list_due_deliveries()
        assert not any(d["id"] == delivery["id"] for d in due)

    async def test_mark_retry_schedules_future_attempt(self):
        from datetime import datetime, timezone

        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        delivery = await repo.enqueue_delivery(
            org_id="org-1", endpoint_id=endpoint["id"], event_type="x", payload={},
        )
        future = "2099-01-01T00:00:00+00:00"
        await repo.mark_retry(delivery["id"], future, "connection refused")

        due_now = await repo.list_due_deliveries()
        assert not any(d["id"] == delivery["id"] for d in due_now)

    async def test_mark_dead_sets_status(self):
        from db.webhook_repo import WebhookRepository

        repo = WebhookRepository()
        endpoint = await repo.create_endpoint(org_id="org-1", url="https://example.com/hook")
        delivery = await repo.enqueue_delivery(
            org_id="org-1", endpoint_id=endpoint["id"], event_type="x", payload={},
        )
        await repo.mark_dead(delivery["id"], "exhausted retries")

        deliveries = await repo.list_deliveries("org-1")
        assert deliveries[0]["status"] == "dead"
