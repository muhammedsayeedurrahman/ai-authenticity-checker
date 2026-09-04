"""Tests for db/compliance_repo.py — content labels and SLA clocks."""

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


SAMPLE_LABEL = {
    "label_code": "synthetically_generated",
    "label_display": "Synthetically generated (AI content)",
    "requires_visible_label": True,
    "risk_score": 0.91,
    "confidence": "HIGH",
    "label_basis": ["Face manipulation detected"],
    "ruleset_version": "in-it-rules-2026.v1",
    "detector_version": "proofyx/learned",
    "assessed_at": "2026-09-04T00:00:00+00:00",
}


class TestContentLabelRepository:
    async def test_create_and_get(self):
        from db.compliance_repo import ContentLabelRepository

        repo = ContentLabelRepository()
        label = await repo.create(
            org_id="org-1", analysis_id="an-1", external_content_ref="post-123",
            label=SAMPLE_LABEL, media_type="image", content_sha256="abc123",
        )
        assert label["label_code"] == "synthetically_generated"
        assert label["label_basis"] == ["Face manipulation detected"]
        assert label["superseded_by_id"] is None

        fetched = await repo.get(label["id"])
        assert fetched == label

    async def test_get_by_external_ref_newest_first(self):
        from db.compliance_repo import ContentLabelRepository

        repo = ContentLabelRepository()
        await repo.create(
            org_id="org-1", analysis_id="an-1", external_content_ref="post-1",
            label=SAMPLE_LABEL,
        )
        await repo.create(
            org_id="org-1", analysis_id="an-2", external_content_ref="post-1",
            label=SAMPLE_LABEL,
        )
        labels = await repo.get_by_external_ref("org-1", "post-1")
        assert len(labels) == 2

    async def test_supersede_links_forward_without_mutating_original(self):
        from db.compliance_repo import ContentLabelRepository

        repo = ContentLabelRepository()
        old = await repo.create(
            org_id="org-1", analysis_id="an-1", external_content_ref="post-2",
            label=SAMPLE_LABEL,
        )
        new = await repo.create(
            org_id="org-1", analysis_id="an-2", external_content_ref="post-2",
            label=SAMPLE_LABEL,
        )
        await repo.supersede(old["id"], new["id"])

        refetched_old = await repo.get(old["id"])
        assert refetched_old["superseded_by_id"] == new["id"]
        assert refetched_old["label_code"] == old["label_code"]  # untouched


class TestSlaRepository:
    async def test_open_clock_computes_due_at(self):
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        clock = await repo.open_clock(
            org_id="org-1", content_label_id="label-1",
            started_at="2026-01-01T00:00:00+00:00", deadline_seconds=10800,
        )
        assert clock["due_at"] == "2026-01-01T03:00:00+00:00"
        assert clock["status"] == "running"

    async def test_close_clock_before_deadline_is_met(self):
        from datetime import datetime, timedelta, timezone

        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        future_due = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        clock = await repo.open_clock(
            org_id="org-1", content_label_id="label-1",
            started_at=datetime.now(timezone.utc).isoformat(), deadline_seconds=3600,
        )
        closed = await repo.close_clock(clock["id"], action="removed", acted_by="mod-1")
        assert closed["status"] == "met"
        assert closed["action"] == "removed"

    async def test_close_clock_after_deadline_is_breached(self):
        from datetime import datetime, timedelta, timezone

        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        past_start = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        clock = await repo.open_clock(
            org_id="org-1", content_label_id="label-1",
            started_at=past_start, deadline_seconds=10800,
        )
        closed = await repo.close_clock(clock["id"], action="removed")
        assert closed["status"] == "breached"

    async def test_close_nonexistent_clock_returns_none(self):
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        assert await repo.close_clock("does-not-exist", action="removed") is None

    async def test_reclosing_an_already_resolved_clock_raises_and_does_not_overwrite(self):
        from datetime import datetime, timedelta, timezone

        from db.compliance_repo import SlaClockAlreadyResolved, SlaRepository

        repo = SlaRepository()
        # Close it while still within the deadline -> "met".
        clock = await repo.open_clock(
            org_id="org-1", content_label_id="label-reclose", deadline_seconds=3600,
        )
        first = await repo.close_clock(clock["id"], action="removed", acted_by="mod-1")
        assert first["status"] == "met"

        # A duplicate/retried call — e.g. a resubmitted webhook — must not
        # silently flip a correct "met" record to "breached" days later.
        with pytest.raises(SlaClockAlreadyResolved):
            await repo.close_clock(clock["id"], action="removed", acted_by="mod-1")

        unchanged = await repo.get(clock["id"])
        assert unchanged["status"] == "met"
        assert unchanged["acted_at"] == first["acted_at"]

    async def test_list_clocks_filters_by_org_and_status(self):
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        c1 = await repo.open_clock(org_id="org-a", content_label_id="l1")
        await repo.open_clock(org_id="org-b", content_label_id="l2")
        await repo.close_clock(c1["id"], action="removed")

        running = await repo.list_clocks("org-a", status="running")
        assert running == []
        all_a = await repo.list_clocks("org-a")
        assert len(all_a) == 1

    async def test_get_by_content_label_id(self):
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        clock = await repo.open_clock(org_id="org-1", content_label_id="label-xyz")
        found = await repo.get_by_content_label_id("label-xyz")
        assert found["id"] == clock["id"]
        assert await repo.get_by_content_label_id("no-such-label") is None

    async def test_mark_notified_sets_correct_column(self):
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        clock = await repo.open_clock(org_id="org-1", content_label_id="l1")
        await repo.mark_notified(clock["id"], "warn")
        refetched = await repo.get(clock["id"])
        assert refetched["warn_notified_at"] is not None
        assert refetched["breach_notified_at"] is None

    async def test_list_due_for_notification_excludes_already_notified(self):
        from db.compliance_repo import SlaRepository

        repo = SlaRepository()
        c1 = await repo.open_clock(org_id="org-1", content_label_id="l1")
        c2 = await repo.open_clock(org_id="org-1", content_label_id="l2")
        await repo.mark_notified(c1["id"], "warn")

        pending = await repo.list_due_for_notification(warn_only=True)
        ids = {c["id"] for c in pending}
        assert c1["id"] not in ids
        assert c2["id"] in ids
