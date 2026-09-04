"""Tests for core/sla_monitor.py's poll pass (run_once)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

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


class TestRunOnce:
    async def test_sends_no_notifications_for_fresh_clock(self):
        from core.sla_monitor import run_once
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        await repo.open_clock(org_id="org-1", content_label_id="l1", deadline_seconds=10800)

        sent = await run_once()
        assert sent == 0

    async def test_sends_warn_notification_past_warn_fraction(self):
        from core.sla_monitor import run_once
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        # 2.5h into a 3h clock — past the 0.67 warn fraction, not yet breached.
        started = (datetime.now(timezone.utc) - timedelta(hours=2, minutes=30)).isoformat()
        clock = await repo.open_clock(
            org_id="org-1", content_label_id="l1", started_at=started, deadline_seconds=10800,
        )

        sent = await run_once()
        assert sent == 1

        refetched = await repo.get(clock["id"])
        assert refetched["warn_notified_at"] is not None
        assert refetched["breach_notified_at"] is None

    async def test_sends_breach_notification_past_deadline(self):
        from core.sla_monitor import run_once
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        started = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        clock = await repo.open_clock(
            org_id="org-1", content_label_id="l1", started_at=started, deadline_seconds=10800,
        )

        sent = await run_once()
        # Both warn (never sent) and breach fire on the same pass — 2 notifications.
        assert sent == 2

        refetched = await repo.get(clock["id"])
        assert refetched["warn_notified_at"] is not None
        assert refetched["breach_notified_at"] is not None

    async def test_idempotent_second_pass_sends_nothing_new(self):
        from core.sla_monitor import run_once
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        started = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        await repo.open_clock(
            org_id="org-1", content_label_id="l1", started_at=started, deadline_seconds=10800,
        )

        first_pass = await run_once()
        second_pass = await run_once()
        assert first_pass == 2
        assert second_pass == 0

    async def test_met_clocks_are_never_notified(self):
        from core.sla_monitor import run_once
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        started = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        clock = await repo.open_clock(
            org_id="org-1", content_label_id="l1", started_at=started, deadline_seconds=10800,
        )
        await repo.close_clock(clock["id"], action="removed")

        sent = await run_once()
        assert sent == 0

    async def test_notification_produces_audit_entry(self):
        from core.sla_monitor import run_once
        from db.audit_log import AuditLog
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        started = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        await repo.open_clock(
            org_id="org-audit", content_label_id="l1", started_at=started, deadline_seconds=10800,
        )

        await run_once()

        entries = await AuditLog().list(org_id="org-audit")
        event_types = {e["event_type"] for e in entries}
        assert "sla.due_soon" in event_types
        assert "sla.breached" in event_types
