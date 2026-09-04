"""
Content-label and SLA-clock repositories for ProofyX compliance features.

Mirrors db/history.py / db/org_repo.py's shape: async SQLAlchemy, plain-
dict returns, a fresh session per call.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy import update as sa_update

from core.sla import compute_due_at, seconds_remaining
from db.compliance_models import ContentLabel, SlaClock
from db.database import get_session_factory

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SlaClockAlreadyResolved(Exception):
    """Raised by SlaRepository.close_clock when the clock is not
    currently "running" — a duplicate/retried action call must never
    silently overwrite an already-resolved compliance record."""

    def __init__(self, current_status: str):
        self.current_status = current_status
        super().__init__(f"SLA clock is already resolved (status={current_status})")


class ContentLabelRepository:
    """Content labels are never updated in place after creation — a
    re-analysis creates a new row and links the old one forward via
    supersede(), so the original determination stays intact."""

    async def create(
        self,
        org_id: str,
        analysis_id: str,
        external_content_ref: str,
        label: dict[str, Any],
        media_type: str = "",
        uploader_ref: str = "",
        content_sha256: str = "",
        verdict: str = "",
    ) -> dict[str, Any]:
        row = ContentLabel(
            id=str(uuid.uuid4()), org_id=org_id, analysis_id=analysis_id,
            external_content_ref=external_content_ref, uploader_ref=uploader_ref,
            media_type=media_type, content_sha256=content_sha256,
            label_code=label.get("label_code", "indeterminate"),
            label_display=label.get("label_display", ""),
            requires_visible_label=bool(label.get("requires_visible_label", False)),
            risk_score=float(label.get("risk_score", 0.0)),
            confidence=label.get("confidence", ""), verdict=verdict,
            label_basis_json=json.dumps(label.get("label_basis", []), default=str),
            ruleset_version=label.get("ruleset_version", ""),
            detector_version=label.get("detector_version", ""),
            labeled_at=label.get("assessed_at") or _now(), created_at=_now(),
        )
        factory = get_session_factory()
        async with factory() as session:
            session.add(row)
            await session.commit()
        return self._to_dict(row)

    async def get(self, label_id: str) -> Optional[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            row = await session.get(ContentLabel, label_id)
            return self._to_dict(row) if row else None

    async def get_by_external_ref(
        self, org_id: str, external_content_ref: str,
    ) -> list[dict[str, Any]]:
        """All labels (including superseded ones) for one platform content
        id, newest first."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = (
                select(ContentLabel)
                .where(
                    ContentLabel.org_id == org_id,
                    ContentLabel.external_content_ref == external_content_ref,
                )
                .order_by(ContentLabel.created_at.desc())
            )
            result = await session.execute(stmt)
            return [self._to_dict(r) for r in result.scalars().all()]

    async def supersede(self, old_label_id: str, new_label_id: str) -> None:
        """Point an old label forward at its replacement. Never mutates
        the old row's own determination fields."""
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                sa_update(ContentLabel)
                .where(ContentLabel.id == old_label_id)
                .values(superseded_by_id=new_label_id)
            )
            await session.commit()

    @staticmethod
    def _to_dict(row: ContentLabel) -> dict[str, Any]:
        return {
            "id": row.id, "org_id": row.org_id, "analysis_id": row.analysis_id,
            "external_content_ref": row.external_content_ref, "uploader_ref": row.uploader_ref,
            "media_type": row.media_type, "content_sha256": row.content_sha256,
            "label_code": row.label_code, "label_display": row.label_display,
            "requires_visible_label": row.requires_visible_label, "risk_score": row.risk_score,
            "confidence": row.confidence, "verdict": row.verdict,
            "label_basis": json.loads(row.label_basis_json or "[]"),
            "ruleset_version": row.ruleset_version, "detector_version": row.detector_version,
            "labeled_at": row.labeled_at, "created_at": row.created_at,
            "superseded_by_id": row.superseded_by_id,
        }


class SlaRepository:
    async def open_clock(
        self,
        org_id: str,
        content_label_id: str,
        analysis_id: str = "",
        started_at: Optional[str] = None,
        deadline_seconds: int = 10800,
        obligation_type: str = "takedown_3h",
    ) -> dict[str, Any]:
        """started_at should be the platform's complaint-receipt time when
        known — see core/sla.py::compute_due_at. Defaults to now."""
        started_at = started_at or _now()
        due_at = compute_due_at(started_at, deadline_seconds)
        row = SlaClock(
            id=str(uuid.uuid4()), org_id=org_id, content_label_id=content_label_id,
            analysis_id=analysis_id, obligation_type=obligation_type,
            started_at=started_at, due_at=due_at, deadline_seconds=deadline_seconds,
            status="running",
        )
        factory = get_session_factory()
        async with factory() as session:
            session.add(row)
            await session.commit()
        return self._to_dict(row)

    async def get(self, clock_id: str) -> Optional[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            row = await session.get(SlaClock, clock_id)
            return self._to_dict(row) if row else None

    async def get_by_content_label_id(self, content_label_id: str) -> Optional[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(SlaClock).where(SlaClock.content_label_id == content_label_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return self._to_dict(row) if row else None

    async def close_clock(
        self, clock_id: str, action: str, acted_by: str = "", notes: str = "",
    ) -> Optional[dict[str, Any]]:
        """Close a running clock. Records "met" or "breached" based on
        whether the deadline had actually passed — the truth, not a
        flattering rounding.

        Raises SlaClockAlreadyResolved if the clock isn't currently
        "running" (already met/breached/cancelled) — the caller must
        treat this as a conflict, not retry-and-overwrite.
        """
        factory = get_session_factory()
        async with factory() as session:
            row = await session.get(SlaClock, clock_id)
            if row is None:
                return None
            if row.status != "running":
                raise SlaClockAlreadyResolved(row.status)
            status = "met" if seconds_remaining(row.due_at) >= 0 else "breached"
            await session.execute(
                sa_update(SlaClock).where(SlaClock.id == clock_id).values(
                    status=status, acted_at=_now(), action=action,
                    acted_by=acted_by, notes=notes,
                )
            )
            await session.commit()
        return await self.get(clock_id)

    async def list_clocks(
        self, org_id: str, status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(SlaClock).where(SlaClock.org_id == org_id)
            if status:
                stmt = stmt.where(SlaClock.status == status)
            stmt = stmt.order_by(SlaClock.due_at.asc())
            result = await session.execute(stmt)
            return [self._to_dict(r) for r in result.scalars().all()]

    async def list_due_for_notification(self, warn_only: bool = False) -> list[dict[str, Any]]:
        """Running clocks not yet notified for the given kind — used by
        core/sla_monitor.py's poll loop."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(SlaClock).where(SlaClock.status == "running")
            if warn_only:
                stmt = stmt.where(SlaClock.warn_notified_at.is_(None))
            else:
                stmt = stmt.where(SlaClock.breach_notified_at.is_(None))
            result = await session.execute(stmt)
            return [self._to_dict(r) for r in result.scalars().all()]

    async def mark_notified(self, clock_id: str, kind: str) -> None:
        """kind: "warn" or "breach"."""
        column = "warn_notified_at" if kind == "warn" else "breach_notified_at"
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                sa_update(SlaClock).where(SlaClock.id == clock_id).values(**{column: _now()})
            )
            await session.commit()

    @staticmethod
    def _to_dict(row: SlaClock) -> dict[str, Any]:
        return {
            "id": row.id, "org_id": row.org_id, "content_label_id": row.content_label_id,
            "analysis_id": row.analysis_id, "obligation_type": row.obligation_type,
            "started_at": row.started_at, "due_at": row.due_at,
            "deadline_seconds": row.deadline_seconds, "status": row.status,
            "acted_at": row.acted_at, "action": row.action, "acted_by": row.acted_by,
            "warn_notified_at": row.warn_notified_at, "breach_notified_at": row.breach_notified_at,
            "notes": row.notes,
        }
