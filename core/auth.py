"""
Supabase JWT authentication for ProofyX.

Validates JWT tokens from Supabase Auth or falls back to API key auth.
When neither SUPABASE_JWT_SECRET nor PROOFYX_API_KEY are configured,
runs in dev mode (unauthenticated).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import jwt
from fastapi import Header, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from core.principal import Principal
from core.secrets import get_pool

logger = logging.getLogger(__name__)

_SUPABASE_JWT_SECRET: Optional[str] = None


def _get_jwt_secret() -> Optional[str]:
    """Load and cache the Supabase JWT secret from environment."""
    global _SUPABASE_JWT_SECRET
    if _SUPABASE_JWT_SECRET is None:
        _SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
    return _SUPABASE_JWT_SECRET or None


def verify_supabase_jwt(token: str) -> dict[str, Any]:
    """Validate a Supabase JWT and return its claims.

    Args:
        token: The raw JWT string (without 'Bearer ' prefix).

    Returns:
        Dict with user claims including 'sub' (user ID) and 'email'.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    secret = _get_jwt_secret()
    if secret is None:
        raise HTTPException(status_code=500, detail="Auth not configured")

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def _resolve_org_for_user(user_id: str, preferred_org_id: Optional[str]) -> Optional[str]:
    """Pick an org_id for a JWT-authenticated user.

    If the caller named an org via X-Proofyx-Org-Id, the user must be a
    member of it. Otherwise falls back to the user's first (oldest-joined)
    org, or None if they belong to none.

    A transient DB error here degrades to org_id=None rather than failing
    the request — org resolution is opportunistic tagging for endpoints
    like /analyze/*, which must keep working even if the org lookup can't
    complete; it must never be the thing that takes down core detection
    because of a database hiccup. This only applies to genuine
    connectivity/query failures, not the deliberate 403 below.
    """
    if not user_id:
        return None
    from db.org_repo import OrgRepository

    repo = OrgRepository()
    if preferred_org_id:
        membership = await repo.get_membership(preferred_org_id, user_id)
        if membership is None:
            raise HTTPException(
                status_code=403, detail="Not a member of the requested organization",
            )
        return preferred_org_id
    try:
        orgs = await repo.list_orgs_for_user(user_id)
    except SQLAlchemyError:
        logger.warning("Org lookup failed for user %s; continuing without org context", user_id)
        return None
    return orgs[0]["id"] if orgs else None


async def _resolve_org_api_key(raw_key: str) -> Optional[dict[str, Any]]:
    """Look up raw_key as an org-scoped API key. Returns None for keys that
    aren't shaped like one (e.g. legacy PROOFYX_API_KEY_* values), so those
    never hit the DB."""
    from db.org_repo import ORG_API_KEY_PREFIX, OrgRepository

    if not raw_key.startswith(ORG_API_KEY_PREFIX):
        return None
    return await OrgRepository().resolve_api_key(raw_key)


async def get_principal(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_proofyx_org_id: Optional[str] = Header(None, alias="X-Proofyx-Org-Id"),
) -> Principal:
    """FastAPI dependency: resolve the full caller identity, including org context.

    Priority:
    1. Bearer JWT token (Supabase Auth) -> Principal(kind="user", org_id=...)
    2. X-API-Key, org-scoped (pfx_live_...) -> Principal(kind="api_key", org_id=...)
    3. X-API-Key, legacy PROOFYX_API_KEY_* -> Principal(kind="api_key", org_id=None)
    4. Dev mode (nothing configured) -> Principal(kind="anonymous", org_id=None)

    Raises:
        HTTPException 401/403/500: same conditions as the legacy
        get_current_user (see its docstring) — this function is the
        single source of truth; get_current_user wraps it.
    """
    jwt_secret = _get_jwt_secret()
    api_key_pool = get_pool("PROOFYX_API_KEY")

    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        if not jwt_secret:
            raise HTTPException(status_code=500, detail="JWT auth not configured on server")
        payload = verify_supabase_jwt(token)
        user_id = payload.get("sub", "")
        email = payload.get("email", "")
        org_id = await _resolve_org_for_user(user_id, x_proofyx_org_id)
        return Principal(kind="user", user_id=user_id, email=email, org_id=org_id)

    if x_api_key:
        org_key = await _resolve_org_api_key(x_api_key)
        if org_key is not None:
            return Principal(
                kind="api_key", org_id=org_key["org_id"], api_key_id=org_key["id"],
                scopes=tuple(org_key.get("scopes", [])),
            )
        if api_key_pool is None:
            raise HTTPException(status_code=403, detail="API key auth not configured")
        if not api_key_pool.has_key(x_api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return Principal(kind="api_key", org_id=None)

    if jwt_secret is None and api_key_pool is None:
        return Principal(kind="anonymous", org_id=None)

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide Authorization: Bearer <token> or X-API-Key header.",
    )


def require_org(principal: Principal) -> str:
    """Return principal.org_id, or raise 403 if the caller has no org context.

    Legacy PROOFYX_API_KEY_* keys and org-less JWT users can still call
    every existing (non-compliance) endpoint — this guard exists only for
    endpoints that need billing/audit tenancy (compliance routes).
    """
    if principal.org_id is None:
        raise HTTPException(
            status_code=403,
            detail=(
                "This endpoint requires an organization-scoped API key or a "
                "JWT user with organization membership; legacy "
                "PROOFYX_API_KEY_* keys are not org-scoped."
            ),
        )
    return principal.org_id


async def get_current_user(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[dict[str, Any]]:
    """FastAPI dependency: authenticate via JWT or API key.

    Thin backward-compatible wrapper over get_principal — preserves the
    exact original contract used throughout api/routes.py:

    Priority:
    1. Bearer JWT token (Supabase Auth) -> returns user dict
    2. X-API-Key header (programmatic access, org-scoped or legacy) ->
       returns None (no user context; endpoints needing org context
       should depend on get_principal directly instead)
    3. Dev mode (nothing configured) -> returns None

    Returns:
        User dict with 'id' and 'email', or None for API key / dev mode.

    Raises:
        HTTPException 401/403/500: If credentials are provided but invalid.
    """
    principal = await get_principal(authorization=authorization, x_api_key=x_api_key)
    if principal.kind == "user":
        return {"id": principal.user_id, "email": principal.email}
    return None
