"""
Background SLA-clock monitor for ProofyX.

Polls for running clocks that have crossed the "due soon" or "breached"
thresholds and haven't been notified yet, fires a webhook + audit entry
for each, and marks them notified (idempotent — a clock is only
notified once per kind). This is a convenience layer only: the
*authoritative* status of any clock is always computed at read time by
core/sla.py::clock_status, so a stopped or crashed monitor only delays a
notification, never produces a wrong answer.

Gated behind PROOFYX_SLA_MONITOR_ENABLED (default "1") so tests that run
the app's real lifespan (tests/conftest.py's `client` fixture) don't leak
a background task — set to "0" there.

Known MVP limitation: with more than one API process/worker, each
instance runs its own poll loop and could send duplicate notifications
for the same clock in a narrow race window. Single-instance deployment
is assumed for now; `SELECT ... FOR UPDATE SKIP LOCKED`-style claiming on
Postgres is the multi-instance hardening (see docs/COMPLIANCE.md).
"""

from __future__ import annotations

import asyncio
import logging
import os

from core.sla import clock_status
from db.audit_log import AuditLog
from db.compliance_repo import SlaRepository

logger = logging.getLogger(__name__)

POLL_SECONDS = int(os.environ.get("PROOFYX_SLA_POLL_SECONDS", "60"))
MONITOR_ENABLED = os.environ.get("PROOFYX_SLA_MONITOR_ENABLED", "1") != "0"


async def _notify(kind: str, clock: dict, sla_repo: SlaRepository, audit_log: AuditLog) -> None:
    from core.webhooks import notify as notify_webhooks

    event_type = "sla.due_soon" if kind == "warn" else "sla.breached"
    payload = {"content_label_id": clock["content_label_id"], "due_at": clock["due_at"]}
    await audit_log.append(
        org_id=clock["org_id"], event_type=event_type, subject_type="sla_clock",
        subject_id=clock["id"], actor_type="system", payload=payload,
    )
    await notify_webhooks(clock["org_id"], event_type, payload)
    await sla_repo.mark_notified(clock["id"], kind)


async def run_once(sla_repo: SlaRepository | None = None, audit_log: AuditLog | None = None) -> int:
    """Run a single poll pass. Returns the number of notifications sent.
    Exposed separately from the loop so tests can drive it deterministically."""
    sla_repo = sla_repo or SlaRepository()
    audit_log = audit_log or AuditLog()
    sent = 0

    for kind, warn_only in (("warn", True), ("breach", False)):
        pending = await sla_repo.list_due_for_notification(warn_only=warn_only)
        for clock in pending:
            effective = clock_status(clock["status"], clock["started_at"], clock["due_at"])
            # A clock that jumps straight from "running" to "breached" (the
            # poll interval missed the due_soon window) must still get its
            # warn notification — "at due_soon or later" is the right test,
            # not "exactly due_soon".
            should_notify = (
                effective in ("due_soon", "breached") if kind == "warn"
                else effective == "breached"
            )
            if not should_notify:
                continue
            await _notify(kind, clock, sla_repo, audit_log)
            sent += 1

    return sent


async def poll_loop(stop_event: asyncio.Event) -> None:
    """Runs until stop_event is set — see main.py's lifespan for start/stop.

    Also drives webhook delivery (core/webhooks.py::process_due_deliveries)
    from the same loop rather than starting a second background task —
    the two chores are both "poll a small due-work queue every minute".
    """
    from core.webhooks import process_due_deliveries

    logger.info("SLA monitor started (poll every %ds)", POLL_SECONDS)
    while not stop_event.is_set():
        try:
            sent = await run_once()
            if sent:
                logger.info("SLA monitor sent %d notification(s)", sent)
        except Exception:
            logger.exception("SLA monitor poll pass failed")
        try:
            delivered = await process_due_deliveries()
            if delivered:
                logger.info("Webhook worker processed %d due deliver(y/ies)", delivered)
        except Exception:
            logger.exception("Webhook delivery poll pass failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=POLL_SECONDS)
        except asyncio.TimeoutError:
            pass
    logger.info("SLA monitor stopped")
