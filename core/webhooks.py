"""
Webhook delivery for ProofyX compliance events.

Signing: Stripe-style — header X-Proofyx-Signature: t=<unix>,v1=<hmac_hex>
over f"{t}.{body}", using each endpoint's own per-endpoint secret.

SSRF defense (mandatory: org-supplied URLs mean this process makes
attacker-influenced outbound requests): https-only, every resolved
address checked against private/loopback/link-local/reserved/CGNAT
ranges (including the 169.254.169.254 cloud-metadata address), no
redirects followed, a bounded timeout, and a capped response read.
deliver() additionally resolves once and connects directly to the
validated IP (not the hostname) to close a DNS-rebinding TOCTOU window —
see that function's docstring. This IP-pinning + Host-header + SNI-
extension approach is the standard mitigation for this class of bug, but
has only been exercised here against mocked httpx clients, not a live
HTTPS endpoint with a real certificate — verify against a real server
before relying on it in production.

Secrets are encrypted at rest (Fernet) — see encrypt_secret/decrypt_secret.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_BYTES = 8192

# Exponential backoff: 1m, 5m, 30m, 2h, 6h — then dead-letter.
RETRY_SCHEDULE_SECONDS = [60, 300, 1800, 7200, 21600]
MAX_ATTEMPTS = len(RETRY_SCHEDULE_SECONDS)

# Auto-deactivate an endpoint after this many consecutive failed deliveries.
DEACTIVATE_AFTER_FAILURES = 10


class WebhookURLRejected(Exception):
    """Raised when a webhook URL fails SSRF validation."""


# RFC 6598 shared/CGNAT address space — not classified as `is_private` by
# Python's ipaddress module, but used internally by some cloud providers
# for pod/service networking, making it a real SSRF-bypass class.
_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")


def _is_disallowed_address(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    if (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    ):
        return True
    return ip.version == 4 and ip in _SHARED_ADDRESS_SPACE


def _resolve_public_addresses(hostname: str) -> list[str]:
    """Resolve hostname and return every address, having verified none of
    them is private/loopback/link-local/multicast/reserved/unspecified/
    CGNAT (this also blocks the 169.254.169.254 cloud-metadata address,
    which falls under link-local). Raises WebhookURLRejected otherwise.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise WebhookURLRejected(f"Could not resolve webhook hostname: {e}")
    if not infos:
        raise WebhookURLRejected("Webhook hostname did not resolve to any address")

    addresses: list[str] = []
    for info in infos:
        addr = info[4][0].split("%", 1)[0]  # strip IPv6 zone id, e.g. "fe80::1%eth0"
        ip = ipaddress.ip_address(addr)
        if _is_disallowed_address(ip):
            raise WebhookURLRejected(f"Webhook URL resolves to a disallowed address: {addr}")
        addresses.append(addr)
    return addresses


def validate_webhook_url(url: str) -> None:
    """Raise WebhookURLRejected if url is unsafe to deliver to.

    Used at registration time (a bad URL is rejected immediately rather
    than discovered at delivery). deliver() performs its own resolve-and-
    pin using the same address-validation rules — see that function's
    docstring for why re-validating here is not itself sufficient to
    prevent DNS-rebinding SSRF.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise WebhookURLRejected("Webhook URL must use https")
    if not parsed.hostname:
        raise WebhookURLRejected("Webhook URL has no hostname")
    _resolve_public_addresses(parsed.hostname)


def sign_payload(secret: str, body: bytes, timestamp: Optional[int] = None) -> str:
    """Stripe-style signature header value: t=<unix>,v1=<hmac_sha256_hex>."""
    ts = timestamp if timestamp is not None else int(time.time())
    material = f"{ts}.".encode("utf-8") + body
    digest = hmac.new(secret.encode("utf-8"), material, hashlib.sha256).hexdigest()
    return f"t={ts},v1={digest}"


def verify_signature(
    secret: str, body: bytes, header_value: str, tolerance_seconds: int = 300,
) -> bool:
    """Reference verifier — the same logic a customer would implement to
    check ProofyX's signature. Rejects stale signatures (replay defense)."""
    try:
        parts = dict(p.split("=", 1) for p in header_value.split(","))
        ts = int(parts["t"])
        given_sig = parts["v1"]
    except (KeyError, ValueError):
        return False
    if abs(time.time() - ts) > tolerance_seconds:
        return False
    expected = sign_payload(secret, body, timestamp=ts).split("v1=", 1)[1]
    return hmac.compare_digest(given_sig, expected)


async def deliver(url: str, secret: str, payload: bytes) -> dict[str, Any]:
    """Attempt one delivery. Never raises — returns
    {"ok", "status", "error"}; retry/dead-letter scheduling is the
    caller's responsibility (see process_due_deliveries).

    SSRF/DNS-rebinding: resolving the hostname and validating it (as
    validate_webhook_url does), then handing the original hostname-based
    URL to httpx, leaves a gap — httpx performs its own independent DNS
    resolution moments later, so a malicious DNS server can serve a public
    IP for the first lookup and a private/internal one for the second
    (classic DNS-rebinding SSRF), bypassing validation entirely. To close
    that window, this function resolves once, validates, and connects
    directly to the validated IP — the hostname is preserved only in the
    Host header and TLS SNI (via httpx's "sni_hostname" request
    extension) so the receiving server and certificate check still see
    the expected name.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return {"ok": False, "status": None, "error": "Webhook URL must use https with a hostname"}

    try:
        addresses = _resolve_public_addresses(parsed.hostname)
    except WebhookURLRejected as e:
        return {"ok": False, "status": None, "error": str(e)}

    resolved_ip = addresses[0]
    ip_netloc = f"[{resolved_ip}]" if ":" in resolved_ip else resolved_ip
    port = parsed.port or 443
    connect_url = urlunparse(parsed._replace(netloc=f"{ip_netloc}:{port}"))

    signature = sign_payload(secret, payload)
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            async with client.stream(
                "POST", connect_url, content=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Proofyx-Signature": signature,
                    "Host": parsed.hostname,
                },
                extensions={"sni_hostname": parsed.hostname},
            ) as resp:
                # Enforce MAX_RESPONSE_BYTES by stopping the read early and
                # closing the connection rather than letting the endpoint
                # stream an unbounded body into memory.
                read_bytes = 0
                async for chunk in resp.aiter_bytes():
                    read_bytes += len(chunk)
                    if read_bytes >= MAX_RESPONSE_BYTES:
                        break
                status_code = resp.status_code
        ok = 200 <= status_code < 300
        return {"ok": ok, "status": status_code, "error": None if ok else f"HTTP {status_code}"}
    except httpx.HTTPError as e:
        return {"ok": False, "status": None, "error": str(e)}


def next_attempt_delay(attempt_number: int) -> Optional[int]:
    """attempt_number is 1-indexed (the attempt that just failed).
    Returns delay seconds before the next retry, or None once attempts
    are exhausted (caller should dead-letter)."""
    if attempt_number > len(RETRY_SCHEDULE_SECONDS):
        return None
    return RETRY_SCHEDULE_SECONDS[attempt_number - 1]


# ── Secret encryption at rest ───────────────────

_fernet: Optional[Fernet] = None
_ephemeral_key_warned = False


def _get_fernet() -> Fernet:
    global _fernet, _ephemeral_key_warned
    if _fernet is None:
        key = os.environ.get("PROOFYX_WEBHOOK_SECRET_KEY", "")
        if not key:
            if not _ephemeral_key_warned:
                logger.warning(
                    "PROOFYX_WEBHOOK_SECRET_KEY not set — using an ephemeral key. "
                    "Webhook secrets will NOT survive a process restart. Generate one "
                    "with Fernet.generate_key() and set it in production."
                )
                _ephemeral_key_warned = True
            key = Fernet.generate_key().decode("utf-8")
        _fernet = Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    return _fernet


def encrypt_secret(raw_secret: str) -> str:
    return _get_fernet().encrypt(raw_secret.encode("utf-8")).decode("utf-8")


def decrypt_secret(encrypted: str) -> str:
    return _get_fernet().decrypt(encrypted.encode("utf-8")).decode("utf-8")


# ── Enqueueing (called from the event-producing side: api/compliance_routes.py,
#    core/sla_monitor.py) ─────────────────────────

async def notify(org_id: str, event_type: str, payload: dict[str, Any]) -> int:
    """Enqueue a delivery for every org endpoint subscribed to event_type.
    Returns the number of deliveries enqueued (0 if the org has no
    matching endpoints — not an error, most orgs won't configure any)."""
    from db.webhook_repo import WebhookRepository  # deferred: avoids import cycle

    repo = WebhookRepository()
    endpoints = await repo.list_active_endpoints_for_event(org_id, event_type)
    for endpoint in endpoints:
        await repo.enqueue_delivery(org_id, endpoint["id"], event_type, payload)
    return len(endpoints)


# ── Delivery worker ──────────────────────────────

async def process_due_deliveries(webhook_repo: Optional[Any] = None) -> int:
    """Process one batch of due webhook deliveries: attempt each, then
    mark delivered / schedule a retry / dead-letter. Returns the number
    of deliveries attempted. Meant to be called from the same poll loop
    as the SLA monitor (core/sla_monitor.py) rather than a second task."""
    from db.webhook_repo import WebhookRepository  # deferred: avoids import cycle

    webhook_repo = webhook_repo or WebhookRepository()
    due = await webhook_repo.list_due_deliveries()
    attempted = 0

    for delivery in due:
        endpoint = await webhook_repo.get_endpoint(delivery["endpoint_id"])
        if endpoint is None or not endpoint["is_active"]:
            await webhook_repo.mark_dead(delivery["id"], "endpoint inactive or missing")
            attempted += 1
            continue

        secret = await webhook_repo.get_endpoint_secret(delivery["endpoint_id"])
        payload_bytes = json.dumps(delivery["payload"], sort_keys=True, default=str).encode("utf-8")
        result = await deliver(endpoint["url"], secret, payload_bytes)
        attempted += 1

        await webhook_repo.record_delivery_result(
            delivery["endpoint_id"], result["ok"], deactivate_after=DEACTIVATE_AFTER_FAILURES,
        )

        if result["ok"]:
            await webhook_repo.mark_delivered(delivery["id"], result["status"] or 0)
            continue

        next_attempt_number = delivery["attempts"] + 1
        delay = next_attempt_delay(next_attempt_number)
        if delay is None:
            await webhook_repo.mark_dead(delivery["id"], result["error"] or "unknown error")
        else:
            next_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
            await webhook_repo.mark_retry(delivery["id"], next_at, result["error"] or "unknown error")

    return attempted
