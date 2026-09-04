"""Tests for db/org_repo.py and core/auth.py's org-aware principal resolution."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "")

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def db_session(monkeypatch):
    """Isolated in-memory SQLite DB with all tables (including compliance
    ones) created — mirrors tests/test_history.py's `db` fixture but
    applies to every test in this module."""
    import db.database as database
    import db.compliance_models  # noqa: F401
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


class TestOrgRepository:
    async def test_create_and_get_org(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme Platform", slug="acme")
        assert org["name"] == "Acme Platform"
        assert org["slug"] == "acme"
        assert org["sla_takedown_seconds"] == 10800

        fetched = await repo.get_org(org["id"])
        assert fetched == org

    async def test_get_org_by_slug(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-2")
        fetched = await repo.get_org_by_slug("acme-2")
        assert fetched["id"] == org["id"]

    async def test_add_member_and_get_membership(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-3")
        await repo.add_member(org["id"], user_id="user-1", role="owner")

        membership = await repo.get_membership(org["id"], "user-1")
        assert membership["role"] == "owner"

        missing = await repo.get_membership(org["id"], "user-2")
        assert missing is None

    async def test_list_orgs_for_user(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org_a = await repo.create_org(name="A", slug="org-a")
        org_b = await repo.create_org(name="B", slug="org-b")
        await repo.add_member(org_a["id"], user_id="user-x", role="owner")
        await repo.add_member(org_b["id"], user_id="user-x", role="viewer")

        orgs = await repo.list_orgs_for_user("user-x")
        assert {o["id"] for o in orgs} == {org_a["id"], org_b["id"]}

    async def test_list_orgs_for_user_with_no_membership(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        assert await repo.list_orgs_for_user("nobody") == []

    async def test_create_org_with_owner_is_atomic(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org_with_owner(
            name="Acme", slug="acme-atomic", owner_user_id="user-owner",
        )
        membership = await repo.get_membership(org["id"], "user-owner")
        assert membership is not None
        assert membership["role"] == "owner"


class TestOrgApiKeys:
    async def test_create_api_key_returns_raw_key_once(self):
        from db.org_repo import ORG_API_KEY_PREFIX, OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-key-1")
        key = await repo.create_api_key(org["id"], created_by_user_id="user-1", label="prod")

        assert key["raw_key"].startswith(ORG_API_KEY_PREFIX)
        assert key["key_prefix"] in key["raw_key"]
        # The raw key itself must never be persisted/returned again.
        listed = await repo.list_api_keys(org["id"])
        assert "raw_key" not in listed[0]

    async def test_resolve_api_key_round_trip(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-key-2")
        key = await repo.create_api_key(org["id"], created_by_user_id="user-1")

        resolved = await repo.resolve_api_key(key["raw_key"])
        assert resolved is not None
        assert resolved["org_id"] == org["id"]
        assert resolved["id"] == key["id"]

    async def test_resolve_unknown_key_returns_none(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        assert await repo.resolve_api_key("pfx_live_doesnotexist") is None

    async def test_revoked_key_no_longer_resolves(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-key-3")
        key = await repo.create_api_key(org["id"], created_by_user_id="user-1")

        revoked = await repo.revoke_api_key(key["id"], org["id"])
        assert revoked is True

        assert await repo.resolve_api_key(key["raw_key"]) is None

    async def test_revoke_nonexistent_key_returns_false(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-key-4")
        assert await repo.revoke_api_key("does-not-exist", org["id"]) is False

    async def test_two_keys_for_same_org_have_distinct_hashes(self):
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-key-5")
        k1 = await repo.create_api_key(org["id"], created_by_user_id="user-1")
        k2 = await repo.create_api_key(org["id"], created_by_user_id="user-1")
        assert k1["raw_key"] != k2["raw_key"]


class TestGetPrincipal:
    async def test_dev_mode_returns_anonymous_principal(self):
        from core.auth import get_principal

        with patch("core.auth._get_jwt_secret", return_value=None), \
             patch("core.auth.get_pool", return_value=None):
            principal = await get_principal(authorization=None, x_api_key=None)
        assert principal.kind == "anonymous"
        assert principal.org_id is None

    async def test_legacy_api_key_yields_no_org(self):
        from core.auth import get_principal
        from core.secrets import KeyPool

        pool = KeyPool("PROOFYX_API_KEY", ["legacy-key-123"])
        with patch("core.auth.get_pool", return_value=pool):
            principal = await get_principal(authorization=None, x_api_key="legacy-key-123")
        assert principal.kind == "api_key"
        assert principal.org_id is None

    async def test_invalid_legacy_api_key_raises_403(self):
        from fastapi import HTTPException

        from core.auth import get_principal
        from core.secrets import KeyPool

        pool = KeyPool("PROOFYX_API_KEY", ["correct-key"])
        with patch("core.auth.get_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                await get_principal(authorization=None, x_api_key="wrong-key")
        assert exc_info.value.status_code == 403

    async def test_org_scoped_api_key_yields_org_id(self):
        from core.auth import get_principal
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-principal-1")
        key = await repo.create_api_key(org["id"], created_by_user_id="user-1")

        with patch("core.auth.get_pool", return_value=None):
            principal = await get_principal(authorization=None, x_api_key=key["raw_key"])
        assert principal.kind == "api_key"
        assert principal.org_id == org["id"]
        assert principal.api_key_id == key["id"]

    async def test_revoked_org_key_falls_through_to_legacy_check(self):
        """A revoked org-shaped key must not silently authenticate — since it
        no longer resolves as an org key, it's checked against the legacy
        pool (and rejected, since it was never a legacy key either)."""
        from fastapi import HTTPException

        from core.auth import get_principal
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-principal-2")
        key = await repo.create_api_key(org["id"], created_by_user_id="user-1")
        await repo.revoke_api_key(key["id"], org["id"])

        with patch("core.auth.get_pool", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await get_principal(authorization=None, x_api_key=key["raw_key"])
        assert exc_info.value.status_code == 403

    async def test_org_lookup_db_error_degrades_to_no_org_context(self):
        """A DB hiccup during opportunistic org resolution must not break
        the request for a JWT user — this is what keeps /analyze/* alive
        even if the org lookup itself is unavailable."""
        from sqlalchemy.exc import SQLAlchemyError

        from core.auth import get_principal

        with patch("core.auth._get_jwt_secret", return_value="test-secret"), \
             patch("core.auth.verify_supabase_jwt", return_value={"sub": "user-1", "email": "u@example.com"}), \
             patch("db.org_repo.OrgRepository.list_orgs_for_user", side_effect=SQLAlchemyError("db down")):
            principal = await get_principal(
                authorization="Bearer faketoken", x_api_key=None, x_proofyx_org_id=None,
            )
        assert principal.kind == "user"
        assert principal.org_id is None

    async def test_missing_credentials_when_auth_configured_raises_401(self):
        from fastapi import HTTPException

        from core.auth import get_principal
        from core.secrets import KeyPool

        pool = KeyPool("PROOFYX_API_KEY", ["some-key"])
        with patch("core.auth.get_pool", return_value=pool):
            with pytest.raises(HTTPException) as exc_info:
                await get_principal(authorization=None, x_api_key=None)
        assert exc_info.value.status_code == 401


class TestRequireOrg:
    async def test_raises_403_when_no_org(self):
        from fastapi import HTTPException

        from core.auth import require_org
        from core.principal import Principal

        with pytest.raises(HTTPException) as exc_info:
            require_org(Principal(kind="api_key", org_id=None))
        assert exc_info.value.status_code == 403

    async def test_returns_org_id_when_present(self):
        from core.auth import require_org
        from core.principal import Principal

        assert require_org(Principal(kind="api_key", org_id="org-123")) == "org-123"


class TestGetCurrentUserBackwardCompatibility:
    """get_current_user must keep its exact old contract even though it now
    wraps get_principal."""

    async def test_org_scoped_api_key_still_returns_none(self):
        from core.auth import get_current_user
        from db.org_repo import OrgRepository

        repo = OrgRepository()
        org = await repo.create_org(name="Acme", slug="acme-gcu-1")
        key = await repo.create_api_key(org["id"], created_by_user_id="user-1")

        with patch("core.auth.get_pool", return_value=None):
            result = await get_current_user(authorization=None, x_api_key=key["raw_key"])
        assert result is None

    async def test_dev_mode_still_returns_none(self):
        from core.auth import get_current_user

        with patch("core.auth._get_jwt_secret", return_value=None), \
             patch("core.auth.get_pool", return_value=None):
            result = await get_current_user(authorization=None, x_api_key=None)
        assert result is None
