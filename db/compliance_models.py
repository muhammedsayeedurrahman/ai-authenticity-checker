"""
Organization/tenant ORM tables for ProofyX compliance features.

Introduces the org/tenant concept the base schema (db/models.py) doesn't
have — needed to scope seat-based billing, org-scoped API keys, webhooks,
and SLA ownership. Shares `Base` with db/models.py so both register on
the same metadata.

IMPORTANT: this module must be imported alongside db.models wherever
Base.metadata is used to create tables (db/database.py::init_db and
alembic/env.py) — otherwise these tables are silently never created.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from db.models import Base


class Organization(Base):
    """A tenant. Owns membership, API keys, and (in later phases) webhooks,
    content labels, and SLA clocks."""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    plan_tier: Mapped[str] = mapped_column(Text, default="trial")
    region: Mapped[str] = mapped_column(Text, default="IN")
    sla_takedown_seconds: Mapped[int] = mapped_column(Integer, default=10800)
    contact_email: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class OrgMember(Base):
    """A user's seat within an organization — the unit seat-based billing counts."""

    __tablename__ = "org_members"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    org_id: Mapped[str] = mapped_column(Text, ForeignKey("organizations.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)  # Supabase JWT "sub"
    # "owner" | "admin" | "compliance_officer" | "viewer" — kept as free text
    # (not a DB enum) so new roles don't require a migration.
    role: Mapped[str] = mapped_column(Text, default="viewer")
    created_at: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("org_id", "user_id", name="uq_org_members_org_user"),
        Index("idx_org_members_user_id", "user_id"),
    )


class OrgApiKey(Base):
    """An org-scoped API key. Only key_hash is persisted — the raw token is
    shown to the caller exactly once, at creation time, and never stored.

    Revocation is a soft `revoked_at` timestamp, never a row delete — a
    revoked key must remain visible in audit/billing history.
    """

    __tablename__ = "org_api_keys"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    org_id: Mapped[str] = mapped_column(Text, ForeignKey("organizations.id"), nullable=False)
    key_prefix: Mapped[str] = mapped_column(Text, nullable=False)  # display only, e.g. "pfx_live_ab12cd34"
    key_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)  # sha256 hex of full token
    label: Mapped[str] = mapped_column(Text, default="")
    scopes: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    created_by_user_id: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    last_used_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    revoked_at: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_org_api_keys_org_id", "org_id"),
    )


class ComplianceAuditLog(Base):
    """Append-only, hash-chained audit trail. See db/audit_log.py for the
    repository — it exposes only append/list/verify_chain, no update or
    delete, ORM-level (see that module's docstring for the honest
    limitation of that boundary).

    `seq` (not `id`) is the primary key and the ordering/chaining key: an
    autoincrementing integer gives strictly monotonic, gap-free ordering
    that a UUID can't guarantee across concurrent inserts.
    """

    __tablename__ = "compliance_audit_log"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    org_id: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[str] = mapped_column(Text, nullable=False)
    actor_type: Mapped[str] = mapped_column(Text, nullable=False)  # system|user|api_key
    actor_id: Mapped[str] = mapped_column(Text, default="")
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    subject_type: Mapped[str] = mapped_column(Text, default="")
    subject_id: Mapped[str] = mapped_column(Text, default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    prev_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    entry_hash: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("idx_audit_org_seq", "org_id", "seq"),
        Index("idx_audit_org_subject", "org_id", "subject_id"),
    )


class ContentLabel(Base):
    """A compliance labeling determination for one piece of platform
    content. Never updated in place after creation — a re-analysis
    creates a new row and sets the old row's superseded_by_id, so the
    original determination stays intact for audit purposes.

    Stores content_sha256, not the media itself, by default — storing raw
    uploaded media plus uploader identifiers would make this an
    erasure-request target under India's DPDP Act, which conflicts with
    an append-only audit trail; media retention is an explicit, later,
    opt-in per org.
    """

    __tablename__ = "content_labels"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    org_id: Mapped[str] = mapped_column(Text, ForeignKey("organizations.id"), nullable=False)
    analysis_id: Mapped[str] = mapped_column(Text, nullable=False)
    external_content_ref: Mapped[str] = mapped_column(Text, nullable=False)  # platform's own content id
    uploader_ref: Mapped[str] = mapped_column(Text, default="")  # opaque, no PII
    media_type: Mapped[str] = mapped_column(Text, default="")
    content_sha256: Mapped[str] = mapped_column(Text, default="")
    label_code: Mapped[str] = mapped_column(Text, nullable=False)
    label_display: Mapped[str] = mapped_column(Text, default="")
    requires_visible_label: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[str] = mapped_column(Text, default="")
    verdict: Mapped[str] = mapped_column(Text, default="")
    label_basis_json: Mapped[str] = mapped_column(Text, default="[]")
    ruleset_version: Mapped[str] = mapped_column(Text, default="")
    detector_version: Mapped[str] = mapped_column(Text, default="")
    labeled_at: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    superseded_by_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_content_labels_org_ref", "org_id", "external_content_ref"),
    )


class SlaClock(Base):
    """A 3-hour (default) takedown-SLA clock for one flagged content label.

    status is written by db/compliance_repo.py::SlaRepository, but the
    *effective* status a caller should trust is always derived at read
    time by core/sla.py::clock_status(status, started_at, due_at) — so a
    stopped background monitor (core/sla_monitor.py) never produces a
    wrong answer, only a late notification.
    """

    __tablename__ = "sla_clocks"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    org_id: Mapped[str] = mapped_column(Text, ForeignKey("organizations.id"), nullable=False)
    content_label_id: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_id: Mapped[str] = mapped_column(Text, default="")
    obligation_type: Mapped[str] = mapped_column(Text, default="takedown_3h")
    started_at: Mapped[str] = mapped_column(Text, nullable=False)
    due_at: Mapped[str] = mapped_column(Text, nullable=False)
    deadline_seconds: Mapped[int] = mapped_column(Integer, default=10800)
    status: Mapped[str] = mapped_column(Text, default="running")  # running|met|breached|cancelled
    acted_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    action: Mapped[str | None] = mapped_column(Text, nullable=True)
    acted_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    warn_notified_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    breach_notified_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (
        Index("idx_sla_org_status_due", "org_id", "status", "due_at"),
    )


class WebhookEndpoint(Base):
    """An org's registered webhook target. The HMAC secret is stored only
    encrypted at rest (core/webhooks.py::encrypt_secret) — the raw secret
    is returned to the caller exactly once, at registration."""

    __tablename__ = "webhook_endpoints"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    org_id: Mapped[str] = mapped_column(Text, ForeignKey("organizations.id"), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    secret_enc: Mapped[str] = mapped_column(Text, nullable=False)
    event_types_json: Mapped[str] = mapped_column(Text, default="[]")  # [] means "all events"
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    last_success_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_failure_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("idx_webhook_endpoints_org", "org_id"),
    )


class WebhookDelivery(Base):
    """One attempted (or pending) delivery of one event to one endpoint.
    status: pending | delivered | failed | dead."""

    __tablename__ = "webhook_deliveries"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    org_id: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint_id: Mapped[str] = mapped_column(Text, ForeignKey("webhook_endpoints.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    delivered_at: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_webhook_deliveries_org", "org_id"),
        Index("idx_webhook_deliveries_status_next", "status", "next_attempt_at"),
    )
