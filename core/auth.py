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


async def get_current_user(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[dict[str, Any]]:
    """FastAPI dependency: authenticate via JWT or API key.

    Priority:
    1. Bearer JWT token (Supabase Auth) -> returns user dict
    2. X-API-Key header (programmatic access) -> returns None (no user context)
    3. Dev mode (nothing configured) -> returns None

    Returns:
        User dict with 'id' and 'email', or None for API key / dev mode.

    Raises:
        HTTPException 401/403: If credentials are provided but invalid.
    """
    jwt_secret = _get_jwt_secret()
    api_key_pool = get_pool("PROOFYX_API_KEY")

    # Try JWT auth first
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        if jwt_secret:
            payload = verify_supabase_jwt(token)
            return {
                "id": payload.get("sub", ""),
                "email": payload.get("email", ""),
            }
        # JWT provided but no secret configured
        raise HTTPException(
            status_code=500,
            detail="JWT auth not configured on server",
        )

    # Try API key auth
    if x_api_key:
        if api_key_pool is None:
            raise HTTPException(status_code=403, detail="API key auth not configured")
        if not api_key_pool.has_key(x_api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return None  # API key auth has no user context

    # Dev mode: no auth configured at all
    if jwt_secret is None and api_key_pool is None:
        return None

    # Auth is configured but no credentials provided
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide Authorization: Bearer <token> or X-API-Key header.",
    )
