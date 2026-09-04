"""
Unified caller identity for ProofyX — spans JWT users, org-scoped API
keys, and legacy unscoped API keys. Built by core.auth.get_principal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    """Resolved identity for the current request.

    kind: "user" (Supabase JWT) | "api_key" (org-scoped or legacy) |
        "anonymous" (dev mode, nothing configured).
    org_id: None when the caller has no organization context — a legacy
        PROOFYX_API_KEY_* caller, a JWT user with no org membership, or
        dev mode. Endpoints that require org scope must call
        core.auth.require_org(principal) rather than assuming it's set.
    """
    kind: str
    user_id: str = ""
    email: str = ""
    org_id: str | None = None
    api_key_id: str | None = None
    scopes: tuple[str, ...] = field(default_factory=tuple)
