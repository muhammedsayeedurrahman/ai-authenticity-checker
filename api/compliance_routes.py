"""
Compliance & traceability API — org/tenant management endpoints.

Phase 2 of the India IT Rules 2026 compliance feature: organizations and
org-scoped API keys. Content labeling, audit trail, and SLA endpoints
land in later phases as this router grows — kept separate from
api/routes.py (already near its size ceiling) rather than folded in.
"""

from __future__ import annotations

import hashlib
import io
import logging
import tempfile
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from api.compliance_schemas import (
    ApiKeyCreatedResponse, ApiKeyListResponse, ApiKeyResponse,
    AuditLogEntry, AuditLogListResponse, ComplianceIngestResponse,
    ComplianceIngestResult, ComplianceLabelResult, CreateApiKeyRequest,
    CreateOrgRequest, CreateWebhookRequest, OrgDetailResponse, OrgListResponse,
    OrgResponse, RevokeApiKeyResponse, RevokeWebhookResponse, SlaActionRequest,
    SlaActionResponse, SlaClockResult, SlaListResponse, TestWebhookResponse,
    WebhookDeliveryListResponse, WebhookDeliveryResponse, WebhookEndpointCreatedResponse,
    WebhookEndpointListResponse, WebhookEndpointResponse,
)
from api.routes import (
    ALLOWED_AUDIO_EXT, ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT,
    MAX_AUDIO_SIZE, MAX_IMAGE_SIZE, MAX_VIDEO_SIZE,
    TIMEOUT_AUDIO, TIMEOUT_IMAGE, TIMEOUT_VIDEO,
    _read_validated, _run_with_timeout, _safe_tmp_remove, limiter,
)
from core.auth import get_principal, require_org
from core.pipeline import analyze_audio, analyze_image, analyze_video
from core.principal import Principal
from core.sla import clock_status
from core.webhooks import WebhookURLRejected
from core.webhooks import deliver as deliver_webhook
from core.webhooks import notify as notify_webhooks
from core.webhooks import validate_webhook_url
from db.audit_log import AuditLog
from db.compliance_repo import ContentLabelRepository, SlaClockAlreadyResolved, SlaRepository
from db.history import AnalysisHistory
from db.org_repo import OrgRepository
from db.webhook_repo import WebhookRepository

logger = logging.getLogger(__name__)

router = APIRouter()
org_repo = OrgRepository()
audit_log = AuditLog()
label_repo = ContentLabelRepository()
sla_repo = SlaRepository()
history = AnalysisHistory()
webhook_repo = WebhookRepository()


def _require_jwt_user(principal: Principal) -> Principal:
    """Org bootstrap requires a real human identity, not an API key."""
    if principal.kind != "user" or not principal.user_id:
        raise HTTPException(
            status_code=401,
            detail="Creating an organization requires a signed-in user (Bearer token), not an API key.",
        )
    return principal


async def _require_org_match(principal: Principal, org_id: str) -> None:
    """Authorize the caller against the org_id in the URL.

    For JWT users this checks membership directly against the URL's
    org_id, independent of get_principal's header-resolved org_id — a
    user belonging to multiple orgs must be able to act on any org they
    are a member of by naming it in the URL path, without also having to
    send X-Proofyx-Org-Id (get_principal otherwise defaults an unheadered
    JWT request to the user's *first-joined* org, which would incorrectly
    403 every other org's management endpoints for that user).

    For API-key principals, org_id is fixed to the key's own org at
    resolution time (core/auth.py::get_principal), so a direct equality
    check is correct and sufficient there.
    """
    if principal.kind == "user":
        if not principal.user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        membership = await org_repo.get_membership(org_id, principal.user_id)
        if membership is None:
            raise HTTPException(status_code=403, detail="Not authorized for this organization")
        return
    if principal.org_id != org_id:
        raise HTTPException(status_code=403, detail="Not authorized for this organization")


# ──────────────────────────────────────────────
# Organizations
# ──────────────────────────────────────────────

@router.post("/orgs", response_model=OrgDetailResponse)
async def create_org(
    body: CreateOrgRequest,
    principal: Principal = Depends(get_principal),
):
    """Create a new organization. The creator becomes its owner."""
    _require_jwt_user(principal)

    existing = await org_repo.get_org_by_slug(body.slug)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Slug '{body.slug}' is already taken")

    org = await org_repo.create_org_with_owner(
        name=body.name, slug=body.slug, owner_user_id=principal.user_id,
        contact_email=body.contact_email,
    )

    return OrgDetailResponse(success=True, data=OrgResponse(**org, role="owner"))


@router.get("/orgs/me", response_model=OrgListResponse)
async def list_my_orgs(principal: Principal = Depends(get_principal)):
    """List organizations the authenticated user belongs to."""
    _require_jwt_user(principal)
    orgs = await org_repo.list_orgs_for_user(principal.user_id)
    return OrgListResponse(success=True, data=[OrgResponse(**o) for o in orgs])


# ──────────────────────────────────────────────
# Org-scoped API keys
# ──────────────────────────────────────────────

@router.post("/orgs/{org_id}/api-keys", response_model=ApiKeyCreatedResponse)
async def create_api_key(
    org_id: str,
    body: CreateApiKeyRequest,
    principal: Principal = Depends(get_principal),
):
    """Issue a new org-scoped API key. The full token is returned exactly
    once, in this response — it cannot be recovered afterward."""
    await _require_org_match(principal, org_id)

    created_by = principal.user_id if principal.kind == "user" else (principal.api_key_id or "")
    key = await org_repo.create_api_key(
        org_id, created_by_user_id=created_by, label=body.label, scopes=body.scopes,
    )
    raw_key = key.pop("raw_key")
    return ApiKeyCreatedResponse(success=True, data=ApiKeyResponse(**key), raw_key=raw_key)


@router.get("/orgs/{org_id}/api-keys", response_model=ApiKeyListResponse)
async def list_api_keys(
    org_id: str,
    principal: Principal = Depends(get_principal),
):
    """List an org's API keys — key_prefix only, never the full token."""
    await _require_org_match(principal, org_id)
    keys = await org_repo.list_api_keys(org_id)
    return ApiKeyListResponse(success=True, data=[ApiKeyResponse(**k) for k in keys])


@router.post("/orgs/{org_id}/api-keys/{key_id}/revoke", response_model=RevokeApiKeyResponse)
async def revoke_api_key(
    org_id: str,
    key_id: str,
    principal: Principal = Depends(get_principal),
):
    """Soft-revoke an API key (POST, not DELETE — this is an audited state
    transition, not a deletion; the key row is kept for billing/audit)."""
    await _require_org_match(principal, org_id)
    revoked = await org_repo.revoke_api_key(key_id, org_id)
    return RevokeApiKeyResponse(success=True, revoked=revoked)


# ──────────────────────────────────────────────
# Content ingestion, labeling, SLA
# ──────────────────────────────────────────────

def _label_response(compliance_label: dict[str, Any]) -> ComplianceLabelResult:
    return ComplianceLabelResult(**{
        k: v for k, v in compliance_label.items() if k in ComplianceLabelResult.model_fields
    })


def _sla_response(clock: dict[str, Any]) -> SlaClockResult:
    effective = clock_status(clock["status"], clock["started_at"], clock["due_at"])
    return SlaClockResult(**clock, effective_status=effective)


def _actor_id(principal: Principal) -> str:
    return principal.user_id or principal.api_key_id or ""


@router.post("/content", response_model=ComplianceIngestResponse)
@limiter.limit("30/minute")
async def ingest_content(
    request: Request,
    file: UploadFile = File(...),
    external_content_ref: str = Form(...),
    media_type: str = Form(..., pattern="^(image|video|audio)$"),
    uploader_ref: str = Form(""),
    flagged_by_complaint: bool = Form(False),
    complaint_received_at: Optional[str] = Form(None),
    principal: Principal = Depends(get_principal),
):
    """Ingest one piece of platform content: run detection, persist a
    compliance label, and — when it's flagged and either tied to a known
    fraud category or explicitly reported via flagged_by_complaint — open
    a takedown-SLA clock. One call produces an analysis, a content_labels
    row, an SLA clock (when applicable), and audit entries.

    complaint_received_at (ISO-8601), when provided, starts the SLA clock
    at the platform's own complaint-receipt time rather than ProofyX's
    scan time — the legally correct clock start (see core/sla.py). A
    complaint already older than the deadline yields an
    immediately-breached clock, which is the honest outcome.
    """
    org_id = require_org(principal)

    if media_type == "image":
        contents = await _read_validated(file, MAX_IMAGE_SIZE, ALLOWED_IMAGE_EXT)
    elif media_type == "video":
        contents = await _read_validated(file, MAX_VIDEO_SIZE, ALLOWED_VIDEO_EXT)
    else:
        contents = await _read_validated(file, MAX_AUDIO_SIZE, ALLOWED_AUDIO_EXT)

    content_sha256 = hashlib.sha256(contents).hexdigest()

    tmp_path = None
    try:
        if media_type == "image":
            try:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
            except (OSError, ValueError, Image.DecompressionBombError):
                raise HTTPException(status_code=400, detail="Invalid image file")
            result = await _run_with_timeout(analyze_image, TIMEOUT_IMAGE, image, mode="ensemble")
        else:
            suffix = ".mp4" if media_type == "video" else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            if media_type == "video":
                result = await _run_with_timeout(analyze_video, TIMEOUT_VIDEO, tmp_path)
            else:
                result = await _run_with_timeout(analyze_audio, TIMEOUT_AUDIO, tmp_path)
    finally:
        _safe_tmp_remove(tmp_path)

    if result.get("error"):
        return ComplianceIngestResponse(success=False, error=result["error"])

    analysis_id = await history.save(
        result, user_id=(principal.user_id if principal.kind == "user" else None), org_id=org_id,
    )

    compliance_label = dict(result.get("compliance_label") or {})
    if flagged_by_complaint and not compliance_label.get("sla_applies"):
        # The caller is telling us a grievance was actually filed — a
        # signal the pipeline itself has no way to infer on its own (see
        # core/compliance_label.py::build_compliance_label).
        from core.compliance_label import build_compliance_label

        compliance_label = build_compliance_label(
            risk_score=compliance_label.get("risk_score", result.get("risk_score", 0.0)),
            confidence=compliance_label.get("confidence", result.get("confidence", "")),
            detector_version=compliance_label.get("detector_version", ""),
            label_basis=compliance_label.get("label_basis", []),
            cybercrime_category=(result.get("cybercrime_risk") or {}).get("category", "none"),
            flagged_by_complaint=True,
        )

    label_row = await label_repo.create(
        org_id=org_id, analysis_id=analysis_id, external_content_ref=external_content_ref,
        label=compliance_label, media_type=media_type, uploader_ref=uploader_ref,
        content_sha256=content_sha256, verdict=result.get("verdict", ""),
    )

    await audit_log.append(
        org_id=org_id, event_type="content.analyzed", subject_type="content_label",
        subject_id=label_row["id"], actor_type=principal.kind, actor_id=_actor_id(principal),
        payload={"analysis_id": analysis_id, "external_content_ref": external_content_ref},
    )
    await audit_log.append(
        org_id=org_id, event_type="content.labeled", subject_type="content_label",
        subject_id=label_row["id"], actor_type="system",
        payload={"label_code": label_row["label_code"], "risk_score": label_row["risk_score"]},
    )
    await notify_webhooks(org_id, "content.labeled", {
        "content_label_id": label_row["id"], "external_content_ref": external_content_ref,
        "label_code": label_row["label_code"],
    })

    sla_result = None
    if compliance_label.get("sla_applies"):
        clock = await sla_repo.open_clock(
            org_id=org_id, content_label_id=label_row["id"], analysis_id=analysis_id,
            started_at=complaint_received_at,
            deadline_seconds=compliance_label.get("sla_deadline_seconds") or 10800,
        )
        await audit_log.append(
            org_id=org_id, event_type="sla.started", subject_type="sla_clock",
            subject_id=clock["id"], actor_type="system",
            payload={"content_label_id": label_row["id"], "due_at": clock["due_at"]},
        )
        await notify_webhooks(org_id, "sla.started", {
            "sla_clock_id": clock["id"], "content_label_id": label_row["id"], "due_at": clock["due_at"],
        })
        sla_result = _sla_response(clock)

    return ComplianceIngestResponse(
        success=True,
        data=ComplianceIngestResult(
            analysis_id=analysis_id, content_label_id=label_row["id"],
            content_sha256=content_sha256, label=_label_response(compliance_label),
            sla=sla_result,
        ),
    )


@router.get("/sla", response_model=SlaListResponse)
async def list_sla_clocks(
    status: Optional[str] = None,
    principal: Principal = Depends(get_principal),
):
    """List an org's SLA clocks. `status` filters on the stored status
    (running/met/breached/cancelled); the response's effective_status
    additionally reinterprets "running" as "due_soon"/"breached" based on
    the current time (see core/sla.py::clock_status)."""
    org_id = require_org(principal)
    clocks = await sla_repo.list_clocks(org_id, status=status)
    return SlaListResponse(success=True, data=[_sla_response(c) for c in clocks])


@router.post("/content/{label_id}/action", response_model=SlaActionResponse)
async def record_content_action(
    label_id: str,
    body: SlaActionRequest,
    principal: Principal = Depends(get_principal),
):
    """Record what the platform did about a flagged piece of content.
    Closes its SLA clock as "met" or "breached" based on whether the
    deadline had actually passed, and appends an audit entry. ProofyX
    never auto-takes-down content — it only records the platform's own
    action."""
    org_id = require_org(principal)

    label = await label_repo.get(label_id)
    if label is None or label["org_id"] != org_id:
        raise HTTPException(status_code=404, detail="Content label not found")

    clock = await sla_repo.get_by_content_label_id(label_id)
    if clock is None:
        raise HTTPException(status_code=404, detail="No SLA clock is open for this content")

    try:
        closed = await sla_repo.close_clock(
            clock["id"], action=body.action, acted_by=body.acted_by or _actor_id(principal),
            notes=body.notes,
        )
    except SlaClockAlreadyResolved as e:
        raise HTTPException(
            status_code=409,
            detail=f"This SLA clock was already resolved (status={e.current_status}); "
                   "a duplicate action call cannot overwrite it.",
        )
    await audit_log.append(
        org_id=org_id, event_type="sla.resolved", subject_type="sla_clock",
        subject_id=closed["id"], actor_type=principal.kind, actor_id=_actor_id(principal),
        payload={"action": body.action, "status": closed["status"]},
    )
    await notify_webhooks(org_id, "sla.resolved", {
        "sla_clock_id": closed["id"], "action": body.action, "status": closed["status"],
    })
    return SlaActionResponse(success=True, data=_sla_response(closed))


@router.get("/audit-log", response_model=AuditLogListResponse)
async def get_audit_log(
    event_type: Optional[str] = None,
    subject_id: Optional[str] = None,
    limit: int = 100,
    principal: Principal = Depends(get_principal),
):
    """Org-scoped audit trail, newest-last (seq order). chain_verified
    reports whether recomputing every entry's hash still matches what's
    stored — a False here means the log was tampered with after the fact."""
    org_id = require_org(principal)
    entries = await audit_log.list(org_id, event_type=event_type, subject_id=subject_id, limit=limit)
    verification = await audit_log.verify_chain(org_id)
    return AuditLogListResponse(
        success=True,
        data=[AuditLogEntry(**e) for e in entries],
        chain_verified=verification["verified"],
    )


# ──────────────────────────────────────────────
# Webhooks
# ──────────────────────────────────────────────

@router.post("/orgs/{org_id}/webhooks", response_model=WebhookEndpointCreatedResponse)
async def create_webhook(
    org_id: str,
    body: CreateWebhookRequest,
    principal: Principal = Depends(get_principal),
):
    """Register a webhook endpoint. The URL is validated against SSRF
    rules (https, public address only) before it's stored — a URL that
    fails validation is rejected at registration, not discovered at
    delivery time. The HMAC secret is returned exactly once."""
    await _require_org_match(principal, org_id)

    try:
        validate_webhook_url(body.url)
    except WebhookURLRejected as e:
        raise HTTPException(status_code=400, detail=str(e))

    endpoint = await webhook_repo.create_endpoint(
        org_id=org_id, url=body.url, event_types=body.event_types,
    )
    secret = endpoint.pop("secret")
    return WebhookEndpointCreatedResponse(
        success=True, data=WebhookEndpointResponse(**endpoint), secret=secret,
    )


@router.get("/orgs/{org_id}/webhooks", response_model=WebhookEndpointListResponse)
async def list_webhooks(
    org_id: str,
    principal: Principal = Depends(get_principal),
):
    """List an org's webhook endpoints — secrets are never included."""
    await _require_org_match(principal, org_id)
    endpoints = await webhook_repo.list_endpoints(org_id)
    return WebhookEndpointListResponse(
        success=True, data=[WebhookEndpointResponse(**e) for e in endpoints],
    )


@router.post("/orgs/{org_id}/webhooks/{endpoint_id}/revoke", response_model=RevokeWebhookResponse)
async def revoke_webhook(
    org_id: str,
    endpoint_id: str,
    principal: Principal = Depends(get_principal),
):
    await _require_org_match(principal, org_id)
    revoked = await webhook_repo.revoke_endpoint(endpoint_id, org_id)
    return RevokeWebhookResponse(success=True, revoked=revoked)


@router.post("/orgs/{org_id}/webhooks/{endpoint_id}/test", response_model=TestWebhookResponse)
async def test_webhook(
    org_id: str,
    endpoint_id: str,
    principal: Principal = Depends(get_principal),
):
    """Send a synchronous test delivery immediately (bypasses the queue)
    so a customer can verify their receiving endpoint and signature
    verification without waiting for a real event."""
    import json as _json

    await _require_org_match(principal, org_id)

    endpoint = await webhook_repo.get_endpoint(endpoint_id)
    if endpoint is None or endpoint["org_id"] != org_id:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")

    secret = await webhook_repo.get_endpoint_secret(endpoint_id)
    payload = _json.dumps(
        {"event_type": "webhook.test", "org_id": org_id}, sort_keys=True,
    ).encode("utf-8")
    result = await deliver_webhook(endpoint["url"], secret, payload)
    return TestWebhookResponse(
        success=True, delivered=result["ok"], status_code=result["status"], error=result["error"],
    )


@router.get("/orgs/{org_id}/webhooks/deliveries", response_model=WebhookDeliveryListResponse)
async def list_webhook_deliveries(
    org_id: str,
    endpoint_id: Optional[str] = None,
    principal: Principal = Depends(get_principal),
):
    await _require_org_match(principal, org_id)
    deliveries = await webhook_repo.list_deliveries(org_id, endpoint_id=endpoint_id)
    return WebhookDeliveryListResponse(
        success=True, data=[WebhookDeliveryResponse(**d) for d in deliveries],
    )
