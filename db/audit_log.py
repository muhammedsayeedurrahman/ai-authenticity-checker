"""
Append-only compliance audit log for ProofyX.

Hash-chained (core/audit_hash.py): each org's entries form their own
chain, so verify_chain can detect tampering with any single row after
the fact. Deliberately exposes only append/list/verify_chain — no
update, no delete method, not even a private one.

Honest limitation: this boundary is enforced at the Python/ORM layer
only. A DB admin with direct SQL access could still UPDATE the table —
the hash chain makes that *detectable* via verify_chain, not
*impossible*. Postgres `BEFORE UPDATE OR DELETE` triggers plus a
restricted DB role are the harder hardening for a later phase; this MVP
ships the detectable version and says so (see docs/COMPLIANCE.md).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select

from core.audit_hash import entry_hash
from db.compliance_models import ComplianceAuditLog
from db.database import get_session_factory

logger = logging.getLogger(__name__)

# Per-org append lock: serializes concurrent appends within this process so
# two requests never read the same "last hash" and race to append. This is
# a single-instance MVP mitigation — running multiple API processes/workers
# needs a DB-level lock (e.g. Postgres SELECT ... FOR UPDATE on a per-org
# cursor row) instead; see docs/COMPLIANCE.md.
_org_locks: dict[str, asyncio.Lock] = {}


def _lock_for(org_id: str) -> asyncio.Lock:
    if org_id not in _org_locks:
        _org_locks[org_id] = asyncio.Lock()
    return _org_locks[org_id]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    """Append-only, hash-chained compliance audit log."""

    async def append(
        self,
        org_id: str,
        event_type: str,
        subject_type: str = "",
        subject_id: str = "",
        actor_type: str = "system",
        actor_id: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        occurred_at = _now()

        async with _lock_for(org_id):
            factory = get_session_factory()
            async with factory() as session:
                stmt = (
                    select(ComplianceAuditLog)
                    .where(ComplianceAuditLog.org_id == org_id)
                    .order_by(ComplianceAuditLog.seq.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                last = result.scalar_one_or_none()
                prev_hash = last.entry_hash if last else None

                new_hash = entry_hash(
                    prev_hash, occurred_at, event_type, subject_id, payload,
                    actor_type=actor_type, actor_id=actor_id, subject_type=subject_type,
                )
                row = ComplianceAuditLog(
                    id=str(uuid.uuid4()), org_id=org_id, occurred_at=occurred_at,
                    actor_type=actor_type, actor_id=actor_id, event_type=event_type,
                    subject_type=subject_type, subject_id=subject_id,
                    payload_json=json.dumps(payload, default=str),
                    prev_hash=prev_hash, entry_hash=new_hash,
                )
                session.add(row)
                await session.commit()
        return self._to_dict(row)

    async def list(
        self,
        org_id: str,
        event_type: Optional[str] = None,
        subject_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(ComplianceAuditLog).where(ComplianceAuditLog.org_id == org_id)
            if event_type:
                stmt = stmt.where(ComplianceAuditLog.event_type == event_type)
            if subject_id:
                stmt = stmt.where(ComplianceAuditLog.subject_id == subject_id)
            if cursor is not None:
                stmt = stmt.where(ComplianceAuditLog.seq > cursor)
            stmt = stmt.order_by(ComplianceAuditLog.seq.asc()).limit(limit)
            result = await session.execute(stmt)
            return [self._to_dict(r) for r in result.scalars().all()]

    async def verify_chain(self, org_id: str) -> dict[str, Any]:
        """Walk an org's chain in seq order, recomputing each entry_hash
        from its stored fields to detect any altered row or broken link."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = (
                select(ComplianceAuditLog)
                .where(ComplianceAuditLog.org_id == org_id)
                .order_by(ComplianceAuditLog.seq.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        prev_hash: Optional[str] = None
        for row in rows:
            payload = json.loads(row.payload_json or "{}")
            expected = entry_hash(
                prev_hash, row.occurred_at, row.event_type, row.subject_id, payload,
                actor_type=row.actor_type, actor_id=row.actor_id, subject_type=row.subject_type,
            )
            if expected != row.entry_hash or row.prev_hash != prev_hash:
                return {"verified": False, "broken_at_seq": row.seq, "entries_checked": len(rows)}
            prev_hash = row.entry_hash

        return {"verified": True, "broken_at_seq": None, "entries_checked": len(rows)}

    @staticmethod
    def _to_dict(row: ComplianceAuditLog) -> dict[str, Any]:
        return {
            "id": row.id, "seq": row.seq, "org_id": row.org_id, "occurred_at": row.occurred_at,
            "actor_type": row.actor_type, "actor_id": row.actor_id, "event_type": row.event_type,
            "subject_type": row.subject_type, "subject_id": row.subject_id,
            "payload": json.loads(row.payload_json or "{}"),
            "prev_hash": row.prev_hash, "entry_hash": row.entry_hash,
        }
