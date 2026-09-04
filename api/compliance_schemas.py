"""Pydantic request/response models for ProofyX compliance endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CreateOrgRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    slug: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    contact_email: str = ""


class OrgResponse(BaseModel):
    id: str
    name: str
    slug: str
    plan_tier: str
    region: str
    sla_takedown_seconds: int
    contact_email: str
    created_at: str
    is_active: bool
    role: Optional[str] = None


class OrgListResponse(BaseModel):
    success: bool
    data: list[OrgResponse] = Field(default_factory=list)


class OrgDetailResponse(BaseModel):
    success: bool
    data: Optional[OrgResponse] = None


class CreateApiKeyRequest(BaseModel):
    label: str = ""
    scopes: list[str] = Field(default_factory=list)


class ApiKeyResponse(BaseModel):
    id: str
    org_id: str
    key_prefix: str
    label: str
    scopes: list[str] = Field(default_factory=list)
    created_by_user_id: str
    created_at: str
    last_used_at: Optional[str] = None
    revoked_at: Optional[str] = None


class ApiKeyCreatedResponse(BaseModel):
    """Returned only once, at creation — raw_key can never be recovered later."""
    success: bool
    data: Optional[ApiKeyResponse] = None
    raw_key: Optional[str] = Field(
        default=None,
        description="Full API token — shown exactly once. Store it now; it cannot be retrieved again.",
    )


class ApiKeyListResponse(BaseModel):
    success: bool
    data: list[ApiKeyResponse] = Field(default_factory=list)


class RevokeApiKeyResponse(BaseModel):
    success: bool
    revoked: bool


# ──────────────────────────────────────────────
# Content ingestion / labeling / SLA
# ──────────────────────────────────────────────

class ComplianceLabelResult(BaseModel):
    label_code: str = "indeterminate"
    label_display: str = ""
    requires_visible_label: bool = False
    requires_embedded_metadata: bool = False
    label_basis: list[str] = Field(default_factory=list)
    regulatory_basis: str = ""
    ruleset_version: str = ""
    detector_version: str = ""
    risk_score: float = 0.0
    confidence: str = ""
    recommended_action: str = "none"
    sla_applies: bool = False
    sla_deadline_seconds: Optional[int] = None
    assessed_at: str = ""
    disclaimer: str = ""


class SlaClockResult(BaseModel):
    id: str
    org_id: str
    content_label_id: str
    analysis_id: str = ""
    obligation_type: str = "takedown_3h"
    started_at: str
    due_at: str
    deadline_seconds: int
    status: str
    effective_status: str
    acted_at: Optional[str] = None
    action: Optional[str] = None
    acted_by: Optional[str] = None
    notes: str = ""


class ComplianceIngestResult(BaseModel):
    analysis_id: str
    content_label_id: str
    content_sha256: str
    label: ComplianceLabelResult
    sla: Optional[SlaClockResult] = None


class ComplianceIngestResponse(BaseModel):
    success: bool
    data: Optional[ComplianceIngestResult] = None
    error: Optional[str] = None


class SlaListResponse(BaseModel):
    success: bool
    data: list[SlaClockResult] = Field(default_factory=list)


class SlaActionRequest(BaseModel):
    action: str = Field(pattern="^(removed|blocked|labeled|restored|cleared_false_positive)$")
    acted_by: str = ""
    notes: str = ""


class SlaActionResponse(BaseModel):
    success: bool
    data: Optional[SlaClockResult] = None


class AuditLogEntry(BaseModel):
    id: str
    seq: int
    org_id: str
    occurred_at: str
    actor_type: str
    actor_id: str
    event_type: str
    subject_type: str
    subject_id: str
    payload: dict = Field(default_factory=dict)
    prev_hash: Optional[str] = None
    entry_hash: str


class AuditLogListResponse(BaseModel):
    success: bool
    data: list[AuditLogEntry] = Field(default_factory=list)
    chain_verified: bool = True


# ──────────────────────────────────────────────
# Webhooks
# ──────────────────────────────────────────────

class CreateWebhookRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    event_types: list[str] = Field(
        default_factory=list,
        description="Subscribed event types; empty means all events.",
    )


class WebhookEndpointResponse(BaseModel):
    id: str
    org_id: str
    url: str
    event_types: list[str] = Field(default_factory=list)
    is_active: bool
    created_at: str
    last_success_at: Optional[str] = None
    last_failure_at: Optional[str] = None
    consecutive_failures: int = 0


class WebhookEndpointCreatedResponse(BaseModel):
    success: bool
    data: Optional[WebhookEndpointResponse] = None
    secret: Optional[str] = Field(
        default=None,
        description="HMAC secret for verifying X-Proofyx-Signature — shown exactly once.",
    )


class WebhookEndpointListResponse(BaseModel):
    success: bool
    data: list[WebhookEndpointResponse] = Field(default_factory=list)


class RevokeWebhookResponse(BaseModel):
    success: bool
    revoked: bool


class TestWebhookResponse(BaseModel):
    success: bool
    delivered: bool
    status_code: Optional[int] = None
    error: Optional[str] = None


class WebhookDeliveryResponse(BaseModel):
    id: str
    org_id: str
    endpoint_id: str
    event_type: str
    status: str
    attempts: int
    next_attempt_at: Optional[str] = None
    response_status: Optional[int] = None
    last_error: Optional[str] = None
    created_at: str
    delivered_at: Optional[str] = None


class WebhookDeliveryListResponse(BaseModel):
    success: bool
    data: list[WebhookDeliveryResponse] = Field(default_factory=list)
