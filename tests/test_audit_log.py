"""Tests for db/audit_log.py — append-only, hash-chained audit trail."""

from __future__ import annotations

import asyncio
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


class TestAppendAndList:
    async def test_append_returns_entry_with_hash(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        entry = await log.append(
            org_id="org-1", event_type="content.labeled", subject_id="label-1",
            payload={"label_code": "synthetically_generated"},
        )
        assert entry["entry_hash"]
        assert entry["prev_hash"] is None
        assert entry["seq"] == 1

    async def test_second_entry_chains_to_first(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        first = await log.append(org_id="org-1", event_type="content.labeled", subject_id="a")
        second = await log.append(org_id="org-1", event_type="sla.started", subject_id="a")
        assert second["prev_hash"] == first["entry_hash"]
        assert second["seq"] == first["seq"] + 1

    async def test_different_orgs_have_independent_chains(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        a1 = await log.append(org_id="org-a", event_type="x", subject_id="s")
        b1 = await log.append(org_id="org-b", event_type="x", subject_id="s")
        assert a1["prev_hash"] is None
        assert b1["prev_hash"] is None

    async def test_list_filters_by_org(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        await log.append(org_id="org-a", event_type="x", subject_id="s")
        await log.append(org_id="org-b", event_type="x", subject_id="s")

        entries = await log.list(org_id="org-a")
        assert len(entries) == 1
        assert entries[0]["org_id"] == "org-a"

    async def test_list_filters_by_event_type_and_subject(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        await log.append(org_id="org-1", event_type="content.labeled", subject_id="s1")
        await log.append(org_id="org-1", event_type="sla.started", subject_id="s1")
        await log.append(org_id="org-1", event_type="content.labeled", subject_id="s2")

        entries = await log.list(org_id="org-1", event_type="content.labeled", subject_id="s1")
        assert len(entries) == 1
        assert entries[0]["subject_id"] == "s1"

    async def test_exposes_no_update_or_delete_method(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        public_methods = {name for name in dir(log) if not name.startswith("_")}
        assert public_methods == {"append", "list", "verify_chain"}


class TestVerifyChain:
    async def test_clean_chain_verifies(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        for i in range(5):
            await log.append(org_id="org-1", event_type=f"event-{i}", subject_id="s")

        result = await log.verify_chain("org-1")
        assert result["verified"] is True
        assert result["broken_at_seq"] is None
        assert result["entries_checked"] == 5

    async def test_empty_chain_verifies_trivially(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        result = await log.verify_chain("org-with-no-entries")
        assert result["verified"] is True
        assert result["entries_checked"] == 0

    async def test_tampered_payload_breaks_chain_at_exact_seq(self):
        from sqlalchemy import update

        from db.audit_log import AuditLog
        from db.compliance_models import ComplianceAuditLog
        from db.database import get_session_factory

        log = AuditLog()
        await log.append(org_id="org-1", event_type="a", subject_id="s")
        second = await log.append(org_id="org-1", event_type="b", subject_id="s")
        await log.append(org_id="org-1", event_type="c", subject_id="s")

        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                update(ComplianceAuditLog)
                .where(ComplianceAuditLog.seq == second["seq"])
                .values(payload_json='{"tampered": true}')
            )
            await session.commit()

        result = await log.verify_chain("org-1")
        assert result["verified"] is False
        assert result["broken_at_seq"] == second["seq"]

    async def test_tampered_actor_id_breaks_chain(self):
        """Regression: actor_id/actor_type must be covered by the hash —
        reassigning who performed an action must be detectable, not just
        tampering with the payload."""
        from sqlalchemy import update

        from db.audit_log import AuditLog
        from db.compliance_models import ComplianceAuditLog
        from db.database import get_session_factory

        log = AuditLog()
        entry = await log.append(
            org_id="org-1", event_type="sla.resolved", subject_id="s",
            actor_type="user", actor_id="employee-1",
        )

        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                update(ComplianceAuditLog)
                .where(ComplianceAuditLog.seq == entry["seq"])
                .values(actor_id="employee-2")
            )
            await session.commit()

        result = await log.verify_chain("org-1")
        assert result["verified"] is False
        assert result["broken_at_seq"] == entry["seq"]

    async def test_concurrent_appends_produce_unbroken_chain(self):
        from db.audit_log import AuditLog

        log = AuditLog()
        await asyncio.gather(*[
            log.append(org_id="org-concurrent", event_type=f"e{i}", subject_id="s")
            for i in range(10)
        ])

        result = await log.verify_chain("org-concurrent")
        assert result["verified"] is True
        assert result["entries_checked"] == 10
