"""
Hash-chain helpers for ProofyX's compliance audit log.

Each entry's hash covers the previous entry's hash plus its own content,
so a row altered in place breaks the chain from that point forward. This
module only computes hashes — db/audit_log.py is the enforcement boundary
(append/list/verify_chain only, no update or delete).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional


def canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON encoding: sorted keys, no extraneous whitespace,
    so the same logical payload always hashes to the same string."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def entry_hash(
    prev_hash: Optional[str],
    occurred_at: str,
    event_type: str,
    subject_id: str,
    payload: dict[str, Any],
    actor_type: str = "",
    actor_id: str = "",
    subject_type: str = "",
) -> str:
    """Sha256 hex digest chaining this entry to the previous one.

    prev_hash is normalized to "" for the chain's first entry so a None
    (first entry) and an empty string (would-be corrupted first entry)
    are never confusable — verify_chain relies on this being unambiguous.

    actor_type/actor_id/subject_type are part of the hashed material so
    that "who did this" and "what kind of thing was acted on" are covered
    by tamper-evidence too — an earlier version only hashed
    event_type/subject_id/payload, which meant a DB-admin edit of
    actor_id (e.g. reassigning a takedown action to a different employee)
    would go undetected by verify_chain even though every other field
    surfaced tampering. actor_type/actor_id default to "" so existing
    callers that don't pass them still hash consistently.
    """
    material = canonical_json({
        "prev_hash": prev_hash or "",
        "occurred_at": occurred_at,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "event_type": event_type,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "payload": payload,
    })
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
