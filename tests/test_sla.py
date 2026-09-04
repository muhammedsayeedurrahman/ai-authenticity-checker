"""Tests for core/sla.py — pure SLA-clock math."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.sla import clock_status, compute_due_at, elapsed_fraction, seconds_remaining


class TestComputeDueAt:
    def test_adds_deadline_seconds(self):
        started = "2026-01-01T00:00:00+00:00"
        due = compute_due_at(started, 10800)
        assert due == "2026-01-01T03:00:00+00:00"

    def test_naive_started_at_treated_as_utc(self):
        due = compute_due_at("2026-01-01T00:00:00", 3600)
        assert "2026-01-01T01:00:00" in due


class TestSecondsRemaining:
    def test_positive_before_deadline(self):
        now = datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        due = "2026-01-01T03:00:00+00:00"
        assert seconds_remaining(due, now) == 7200

    def test_negative_after_deadline(self):
        now = datetime(2026, 1, 1, 4, 0, 0, tzinfo=timezone.utc)
        due = "2026-01-01T03:00:00+00:00"
        assert seconds_remaining(due, now) == -3600

    def test_zero_exactly_at_deadline(self):
        now = datetime(2026, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
        due = "2026-01-01T03:00:00+00:00"
        assert seconds_remaining(due, now) == 0


class TestElapsedFraction:
    def test_zero_at_start(self):
        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert elapsed_fraction("2026-01-01T00:00:00+00:00", "2026-01-01T03:00:00+00:00", now) == 0.0

    def test_one_at_due(self):
        now = datetime(2026, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
        assert elapsed_fraction("2026-01-01T00:00:00+00:00", "2026-01-01T03:00:00+00:00", now) == 1.0

    def test_clamped_past_due(self):
        now = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert elapsed_fraction("2026-01-01T00:00:00+00:00", "2026-01-01T03:00:00+00:00", now) == 1.0

    def test_half_way(self):
        now = datetime(2026, 1, 1, 1, 30, 0, tzinfo=timezone.utc)
        frac = elapsed_fraction("2026-01-01T00:00:00+00:00", "2026-01-01T03:00:00+00:00", now)
        assert abs(frac - 0.5) < 1e-9


class TestClockStatus:
    started = "2026-01-01T00:00:00+00:00"
    due = "2026-01-01T03:00:00+00:00"

    def test_terminal_statuses_pass_through_unchanged(self):
        for terminal in ("met", "breached", "cancelled"):
            now = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
            assert clock_status(terminal, self.started, self.due, now) == terminal

    def test_running_early_stays_running(self):
        now = datetime(2026, 1, 1, 0, 30, 0, tzinfo=timezone.utc)
        assert clock_status("running", self.started, self.due, now) == "running"

    def test_running_past_warn_fraction_is_due_soon(self):
        # 0.67 of 3h = ~2h into the clock
        now = datetime(2026, 1, 1, 2, 15, 0, tzinfo=timezone.utc)
        assert clock_status("running", self.started, self.due, now) == "due_soon"

    def test_running_past_deadline_is_breached(self):
        now = datetime(2026, 1, 1, 4, 0, 0, tzinfo=timezone.utc)
        assert clock_status("running", self.started, self.due, now) == "breached"

    def test_backdated_complaint_already_breached_at_ingestion(self):
        """A complaint reported as received 4 hours ago against a 3-hour
        SLA must read as breached immediately, not 'running'."""
        four_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
        due_at = compute_due_at(four_hours_ago, 10800)
        assert clock_status("running", four_hours_ago, due_at) == "breached"
