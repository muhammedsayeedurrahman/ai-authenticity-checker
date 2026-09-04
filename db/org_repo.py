"""
Organization/membership/API-key repository for ProofyX.

Async SQLAlchemy, mirrors db/history.py's shape: plain-dict returns,
a fresh get_session_factory() session per call. API keys are hashed at
rest (sha256) — the raw token is returned to the caller exactly once,
at creation time, and is never persisted or logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy import update as sa_update

from db.compliance_models import Organization, OrgApiKey, OrgMember
from db.database import get_session_factory

logger = logging.getLogger(__name__)

ORG_API_KEY_PREFIX = "pfx_live_"
_KEY_DISPLAY_CHARS = 8  # chars of the random portion shown in key_prefix


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class OrgRepository:
    """Async SQLAlchemy-backed organization/membership/API-key repository."""

    # ── Organizations ──────────────────────────

    async def create_org(
        self, name: str, slug: str, contact_email: str = "",
        sla_takedown_seconds: int = 10800,
    ) -> dict[str, Any]:
        org = Organization(
            id=str(uuid.uuid4()), name=name, slug=slug,
            contact_email=contact_email, sla_takedown_seconds=sla_takedown_seconds,
            created_at=_now(),
        )
        factory = get_session_factory()
        async with factory() as session:
            session.add(org)
            await session.commit()
        return self._org_to_dict(org)

    async def create_org_with_owner(
        self, name: str, slug: str, owner_user_id: str, contact_email: str = "",
        sla_takedown_seconds: int = 10800,
    ) -> dict[str, Any]:
        """Create an org and its first membership row in one transaction.

        Without this, create_org() succeeding followed by a failed
        add_member() (DB blip, pool exhaustion) would leave an org with
        zero members, permanently unmanageable — there is no invite/add-
        member endpoint, and the slug becomes permanently unavailable
        since get_org_by_slug still finds it.
        """
        org = Organization(
            id=str(uuid.uuid4()), name=name, slug=slug,
            contact_email=contact_email, sla_takedown_seconds=sla_takedown_seconds,
            created_at=_now(),
        )
        member = OrgMember(
            id=str(uuid.uuid4()), org_id=org.id, user_id=owner_user_id, role="owner",
            created_at=_now(),
        )
        factory = get_session_factory()
        async with factory() as session:
            session.add(org)
            session.add(member)
            await session.commit()
        return self._org_to_dict(org)

    async def get_org(self, org_id: str) -> Optional[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            row = await session.get(Organization, org_id)
            return self._org_to_dict(row) if row else None

    async def get_org_by_slug(self, slug: str) -> Optional[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(Organization).where(Organization.slug == slug)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return self._org_to_dict(row) if row else None

    async def list_orgs_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """Orgs a user belongs to, each annotated with their role, ordered by
        join time (oldest first) — used to pick a default org when the
        caller doesn't specify one."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = (
                select(Organization, OrgMember.role)
                .join(OrgMember, OrgMember.org_id == Organization.id)
                .where(OrgMember.user_id == user_id)
                .order_by(OrgMember.created_at.asc())
            )
            result = await session.execute(stmt)
            return [{**self._org_to_dict(org), "role": role} for org, role in result.all()]

    # ── Membership ──────────────────────────────

    async def add_member(self, org_id: str, user_id: str, role: str = "owner") -> dict[str, Any]:
        member = OrgMember(
            id=str(uuid.uuid4()), org_id=org_id, user_id=user_id, role=role, created_at=_now(),
        )
        factory = get_session_factory()
        async with factory() as session:
            session.add(member)
            await session.commit()
        return self._member_to_dict(member)

    async def get_membership(self, org_id: str, user_id: str) -> Optional[dict[str, Any]]:
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(OrgMember).where(
                OrgMember.org_id == org_id, OrgMember.user_id == user_id,
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return self._member_to_dict(row) if row else None

    # ── API keys ────────────────────────────────

    async def create_api_key(
        self, org_id: str, created_by_user_id: str, label: str = "",
        scopes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Create a new org-scoped API key.

        Returns the usual key dict plus "raw_key" — the full token, present
        ONLY in this return value. Only its sha256 hash is ever persisted;
        callers must display/copy raw_key immediately, it cannot be
        recovered later.
        """
        random_part = secrets.token_urlsafe(32)
        raw_key = f"{ORG_API_KEY_PREFIX}{random_part}"
        key_prefix = raw_key[: len(ORG_API_KEY_PREFIX) + _KEY_DISPLAY_CHARS]

        row = OrgApiKey(
            id=str(uuid.uuid4()), org_id=org_id, key_prefix=key_prefix,
            key_hash=_hash_key(raw_key), label=label,
            scopes=json.dumps(scopes or []), created_by_user_id=created_by_user_id,
            created_at=_now(),
        )
        factory = get_session_factory()
        async with factory() as session:
            session.add(row)
            await session.commit()

        result = self._api_key_to_dict(row)
        result["raw_key"] = raw_key
        return result

    async def resolve_api_key(self, raw_key: str) -> Optional[dict[str, Any]]:
        """Look up an org API key by its raw token. Excludes revoked keys.

        Updates last_used_at as a side effect (best-effort — failures here
        must never block the caller's actual request).
        """
        factory = get_session_factory()
        key_hash = _hash_key(raw_key)
        async with factory() as session:
            stmt = select(OrgApiKey).where(
                OrgApiKey.key_hash == key_hash, OrgApiKey.revoked_at.is_(None),
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            key_dict = self._api_key_to_dict(row)
            try:
                await session.execute(
                    sa_update(OrgApiKey).where(OrgApiKey.id == row.id).values(last_used_at=_now())
                )
                await session.commit()
            except Exception:
                logger.warning("Failed to update last_used_at for API key %s", row.id)
            return key_dict

    async def revoke_api_key(self, key_id: str, org_id: str) -> bool:
        """Soft-revoke an org API key (sets revoked_at). Never deletes the row."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = (
                sa_update(OrgApiKey)
                .where(OrgApiKey.id == key_id, OrgApiKey.org_id == org_id)
                .values(revoked_at=_now())
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def list_api_keys(self, org_id: str) -> list[dict[str, Any]]:
        """List keys for an org. key_hash is never exposed, only key_prefix."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(OrgApiKey).where(OrgApiKey.org_id == org_id)
            result = await session.execute(stmt)
            return [self._api_key_to_dict(r) for r in result.scalars().all()]

    # ── dict conversion ─────────────────────────

    @staticmethod
    def _org_to_dict(row: Organization) -> dict[str, Any]:
        return {
            "id": row.id, "name": row.name, "slug": row.slug,
            "plan_tier": row.plan_tier, "region": row.region,
            "sla_takedown_seconds": row.sla_takedown_seconds,
            "contact_email": row.contact_email, "created_at": row.created_at,
            "is_active": row.is_active,
        }

    @staticmethod
    def _member_to_dict(row: OrgMember) -> dict[str, Any]:
        return {
            "id": row.id, "org_id": row.org_id, "user_id": row.user_id,
            "role": row.role, "created_at": row.created_at,
        }

    @staticmethod
    def _api_key_to_dict(row: OrgApiKey) -> dict[str, Any]:
        return {
            "id": row.id, "org_id": row.org_id, "key_prefix": row.key_prefix,
            "label": row.label, "scopes": json.loads(row.scopes or "[]"),
            "created_by_user_id": row.created_by_user_id, "created_at": row.created_at,
            "last_used_at": row.last_used_at, "revoked_at": row.revoked_at,
        }
