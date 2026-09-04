"""
Pure SLA-clock math for ProofyX compliance takedown obligations.

No DB, no I/O. Effective status is always derived at read time from
due_at vs now — the DB row's stored `status` is authoritative only once
it reaches a terminal state (met/breached/cancelled); while "running" it
is reinterpreted here. This means a stopped/crashed background monitor
(core/sla_monitor.py) can only delay a notification, never produce a
wrong status.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

# Fraction of the deadline elapsed before a running clock is considered
# "due soon" (e.g. 0.67 of a 3-hour clock = 2 hours elapsed, 1 remaining).
WARN_FRACTION = 0.67


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute_due_at(started_at: str, deadline_seconds: int) -> str:
    """The takedown deadline, deadline_seconds after started_at.

    started_at should be the platform's complaint-receipt time when known
    (not ProofyX's scan time) — see db/compliance_repo.py::SlaRepository.open_clock.
    """
    start = _parse(started_at)
    return (start + timedelta(seconds=deadline_seconds)).isoformat()


def seconds_remaining(due_at: str, now: Optional[datetime] = None) -> float:
    """Negative once the deadline has passed."""
    now = now or datetime.now(timezone.utc)
    return (_parse(due_at) - now).total_seconds()


def elapsed_fraction(started_at: str, due_at: str, now: Optional[datetime] = None) -> float:
    """0.0 at start, 1.0 at (or past) due_at. Clamped to [0, 1]."""
    now = now or datetime.now(timezone.utc)
    start = _parse(started_at)
    due = _parse(due_at)
    total = (due - start).total_seconds()
    if total <= 0:
        return 1.0
    elapsed = (now - start).total_seconds()
    return max(0.0, min(1.0, elapsed / total))


def clock_status(
    status: str, started_at: str, due_at: str, now: Optional[datetime] = None,
) -> str:
    """Effective status for display/alerting.

    A stored "running" status is reinterpreted as "due_soon" or "breached"
    based on the current time; terminal statuses (met/breached/cancelled)
    pass through unchanged since they were already resolved by an actual
    action (db/compliance_repo.py::SlaRepository.close_clock).
    """
    if status != "running":
        return status
    if seconds_remaining(due_at, now) < 0:
        return "breached"
    if elapsed_fraction(started_at, due_at, now) >= WARN_FRACTION:
        return "due_soon"
    return "running"
