"""
Webhook endpoint and delivery repository for ProofyX compliance events.

Mirrors the other repositories' shape: async SQLAlchemy, plain-dict
returns, a fresh session per call.
"""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy import update as sa_update

from core.webhooks import decrypt_secret, encrypt_secret
from db.compliance_models import WebhookDelivery, WebhookEndpoint
from db.database import get_session_factory

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WebhookRepository:
    # ── Endpoints ───────────────────────────────

    async def create_endpoint(
        self, org_id: str, url: str, event_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Returns the endpoint dict plus "secret" — the raw HMAC secret,
        present ONLY in this return value (same one-time-reveal pattern as
        db/org_repo.py::create_api_key)."""
        raw_secret = secrets.token_urlsafe(32)
        row = WebhookEndpoint(
            id=str(uuid.uuid4()), org_id=org_id, url=url,
            secret_enc=encrypt_secret(raw_secret),
            event_types_json=json.dumps(event_types or []),
            created_at=_now(),
        )
        factory = get_session_factory()
        async with factory() as session:
            session.add(row)
            await session.commit()
        result = self._endpoint_to_dict(row)
        result["secret"] = raw_secret
        return result

    async def get_endpoint(self, endpoint_id: str) -> Optional[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            row = await session.get(WebhookEndpoint, endpoint_id)
            return self._endpoint_to_dict(row) if row else None

    async def get_endpoint_secret(self, endpoint_id: str) -> Optional[str]:
        factory = get_session_factory()
        async with factory() as session:
            row = await session.get(WebhookEndpoint, endpoint_id)
            return decrypt_secret(row.secret_enc) if row else None

    async def list_endpoints(self, org_id: str) -> list[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(WebhookEndpoint).where(WebhookEndpoint.org_id == org_id)
            result = await session.execute(stmt)
            return [self._endpoint_to_dict(r) for r in result.scalars().all()]

    async def list_active_endpoints_for_event(
        self, org_id: str, event_type: str,
    ) -> list[dict[str, Any]]:
        """Active endpoints subscribed to event_type (an empty
        event_types list on an endpoint means "all events")."""
        endpoints = await self.list_endpoints(org_id)
        return [
            e for e in endpoints
            if e["is_active"] and (not e["event_types"] or event_type in e["event_types"])
        ]

    async def revoke_endpoint(self, endpoint_id: str, org_id: str) -> bool:
        factory = get_session_factory()
        async with factory() as session:
            stmt = (
                sa_update(WebhookEndpoint)
                .where(WebhookEndpoint.id == endpoint_id, WebhookEndpoint.org_id == org_id)
                .values(is_active=False)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def record_delivery_result(
        self, endpoint_id: str, ok: bool, deactivate_after: int = 10,
    ) -> None:
        """Update endpoint health counters after a delivery attempt;
        auto-deactivate after too many consecutive failures."""
        factory = get_session_factory()
        async with factory() as session:
            row = await session.get(WebhookEndpoint, endpoint_id)
            if row is None:
                return
            if ok:
                await session.execute(
                    sa_update(WebhookEndpoint).where(WebhookEndpoint.id == endpoint_id).values(
                        last_success_at=_now(), consecutive_failures=0,
                    )
                )
            else:
                new_failures = row.consecutive_failures + 1
                values: dict[str, Any] = {
                    "last_failure_at": _now(), "consecutive_failures": new_failures,
                }
                if new_failures >= deactivate_after:
                    values["is_active"] = False
                await session.execute(
                    sa_update(WebhookEndpoint).where(WebhookEndpoint.id == endpoint_id).values(**values)
                )
            await session.commit()

    # ── Deliveries ──────────────────────────────

    async def enqueue_delivery(
        self, org_id: str, endpoint_id: str, event_type: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        row = WebhookDelivery(
            id=str(uuid.uuid4()), org_id=org_id, endpoint_id=endpoint_id,
            event_type=event_type, payload_json=json.dumps(payload, default=str),
            status="pending", attempts=0, next_attempt_at=_now(), created_at=_now(),
        )
        factory = get_session_factory()
        async with factory() as session:
            session.add(row)
            await session.commit()
        return self._delivery_to_dict(row)

    async def list_due_deliveries(self, limit: int = 50) -> list[dict[str, Any]]:
        factory = get_session_factory()
        now = _now()
        async with factory() as session:
            stmt = (
                select(WebhookDelivery)
                .where(WebhookDelivery.status == "pending", WebhookDelivery.next_attempt_at <= now)
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [self._delivery_to_dict(r) for r in result.scalars().all()]

    async def mark_delivered(self, delivery_id: str, response_status: int) -> None:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                sa_update(WebhookDelivery).where(WebhookDelivery.id == delivery_id).values(
                    status="delivered", delivered_at=_now(), response_status=response_status,
                    attempts=WebhookDelivery.attempts + 1,
                )
            )
            await session.commit()

    async def mark_retry(self, delivery_id: str, next_attempt_at: str, error: str) -> None:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                sa_update(WebhookDelivery).where(WebhookDelivery.id == delivery_id).values(
                    next_attempt_at=next_attempt_at, last_error=error,
                    attempts=WebhookDelivery.attempts + 1,
                )
            )
            await session.commit()

    async def mark_dead(self, delivery_id: str, error: str) -> None:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                sa_update(WebhookDelivery).where(WebhookDelivery.id == delivery_id).values(
                    status="dead", last_error=error, attempts=WebhookDelivery.attempts + 1,
                )
            )
            await session.commit()

    async def list_deliveries(
        self, org_id: str, endpoint_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(WebhookDelivery).where(WebhookDelivery.org_id == org_id)
            if endpoint_id:
                stmt = stmt.where(WebhookDelivery.endpoint_id == endpoint_id)
            stmt = stmt.order_by(WebhookDelivery.created_at.desc())
            result = await session.execute(stmt)
            return [self._delivery_to_dict(r) for r in result.scalars().all()]

    @staticmethod
    def _endpoint_to_dict(row: WebhookEndpoint) -> dict[str, Any]:
        return {
            "id": row.id, "org_id": row.org_id, "url": row.url,
            "event_types": json.loads(row.event_types_json or "[]"),
            "is_active": row.is_active, "created_at": row.created_at,
            "last_success_at": row.last_success_at, "last_failure_at": row.last_failure_at,
            "consecutive_failures": row.consecutive_failures,
        }

    @staticmethod
    def _delivery_to_dict(row: WebhookDelivery) -> dict[str, Any]:
        return {
            "id": row.id, "org_id": row.org_id, "endpoint_id": row.endpoint_id,
            "event_type": row.event_type, "payload": json.loads(row.payload_json or "{}"),
            "status": row.status, "attempts": row.attempts, "next_attempt_at": row.next_attempt_at,
            "response_status": row.response_status, "last_error": row.last_error,
            "created_at": row.created_at, "delivered_at": row.delivered_at,
        }
