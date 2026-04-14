"""Tests for core.secrets — KeyPool round-robin and cooldown logic."""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import patch

import pytest

from core.secrets import AllKeysCooledDown, KeyPool, load_key_pool


# ──────────────────────────────────────────────
# KeyPool — construction
# ──────────────────────────────────────────────

class TestKeyPoolConstruction:
    def test_requires_at_least_one_key(self):
        with pytest.raises(ValueError, match="at least one key"):
            KeyPool("test_svc", [])

    def test_size_property(self):
        pool = KeyPool("svc", ["a", "b", "c"])
        assert pool.size == 3

    def test_repr_includes_service_name(self):
        pool = KeyPool("my_service", ["k1"])
        assert "my_service" in repr(pool)


# ──────────────────────────────────────────────
# KeyPool — round-robin
# ──────────────────────────────────────────────

class TestRoundRobin:
    def test_single_key_always_returns_same(self):
        pool = KeyPool("svc", ["only_key"])
        for _ in range(5):
            assert pool.get_key() == "only_key"

    def test_cycles_through_keys(self):
        keys = ["a", "b", "c"]
        pool = KeyPool("svc", keys)
        result = [pool.get_key() for _ in range(6)]
        assert result == ["a", "b", "c", "a", "b", "c"]

    def test_available_count_starts_at_total(self):
        pool = KeyPool("svc", ["a", "b", "c"])
        assert pool.available_count() == 3


# ──────────────────────────────────────────────
# KeyPool — cooldown
# ──────────────────────────────────────────────

class TestCooldown:
    def test_failed_key_is_skipped(self):
        pool = KeyPool("svc", ["a", "b", "c"])
        pool.get_key()  # returns "a", advances index
        pool.report_failure("a", cooldown_sec=60)

        # Next calls should skip "a"
        results = [pool.get_key() for _ in range(4)]
        assert "a" not in results
        assert set(results) == {"b", "c"}

    def test_available_count_decreases_on_failure(self):
        pool = KeyPool("svc", ["a", "b", "c"])
        pool.report_failure("a", cooldown_sec=60)
        assert pool.available_count() == 2

    def test_all_keys_cooled_down_raises(self):
        pool = KeyPool("svc", ["a", "b"])
        pool.report_failure("a", cooldown_sec=60)
        pool.report_failure("b", cooldown_sec=60)

        with pytest.raises(AllKeysCooledDown) as exc_info:
            pool.get_key()
        assert exc_info.value.service == "svc"
        assert exc_info.value.retry_after > 0

    def test_cooldown_expires(self):
        pool = KeyPool("svc", ["a", "b"])
        pool.report_failure("a", cooldown_sec=0.1)  # 100ms cooldown

        # "a" is skipped initially
        assert pool.get_key() == "b"

        # Wait for cooldown to expire
        time.sleep(0.15)

        # "a" should be available again
        keys = {pool.get_key() for _ in range(4)}
        assert "a" in keys

    def test_report_success_clears_cooldown(self):
        pool = KeyPool("svc", ["a", "b"])
        pool.report_failure("a", cooldown_sec=300)
        assert pool.available_count() == 1

        pool.report_success("a")
        assert pool.available_count() == 2

    def test_report_failure_unknown_key_is_noop(self):
        pool = KeyPool("svc", ["a"])
        pool.report_failure("unknown_key")  # should not raise
        assert pool.available_count() == 1

    def test_report_success_unknown_key_is_noop(self):
        pool = KeyPool("svc", ["a"])
        pool.report_success("unknown_key")  # should not raise


# ──────────────────────────────────────────────
# KeyPool — thread safety
# ──────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_get_key(self):
        pool = KeyPool("svc", ["a", "b", "c"])
        results: list[str] = []
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(100):
                    results.append(pool.get_key())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 400
        assert set(results) == {"a", "b", "c"}


# ──────────────────────────────────────────────
# load_key_pool — env var loading
# ──────────────────────────────────────────────

class TestLoadKeyPool:
    def test_returns_none_when_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            result = load_key_pool("NONEXISTENT_SVC")
            assert result is None

    def test_loads_numbered_keys(self):
        env = {
            "MY_SVC_1": "key_one",
            "MY_SVC_2": "key_two",
            "MY_SVC_3": "key_three",
        }
        with patch.dict(os.environ, env, clear=True):
            pool = load_key_pool("MY_SVC")
            assert pool is not None
            assert pool.size == 3
            assert pool.service == "MY_SVC"

    def test_stops_at_gap(self):
        env = {
            "MY_SVC_1": "key_one",
            "MY_SVC_2": "key_two",
            # _3 is missing
            "MY_SVC_4": "key_four",
        }
        with patch.dict(os.environ, env, clear=True):
            pool = load_key_pool("MY_SVC")
            assert pool is not None
            assert pool.size == 2  # stops at gap

    def test_skips_empty_values(self):
        env = {
            "MY_SVC_1": "key_one",
            "MY_SVC_2": "   ",  # whitespace only
            "MY_SVC_3": "key_three",
        }
        with patch.dict(os.environ, env, clear=True):
            pool = load_key_pool("MY_SVC")
            assert pool is not None
            # _2 is empty/whitespace so skipped, but _3 never reached (gap at _2)
            # Actually load_key_pool skips empty but doesn't break — let me re-check
            # The implementation breaks on None (missing), not empty
            # _2 exists but is whitespace → stripped to "" → skipped (not appended)
            # _3 exists → appended
            assert pool.size == 2  # "key_one" and "key_three"


# ──────────────────────────────────────────────
# AllKeysCooledDown exception
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# KeyPool — has_key (constant-time)
# ──────────────────────────────────────────────

class TestHasKey:
    def test_returns_true_for_existing_key(self):
        pool = KeyPool("svc", ["alpha", "beta"])
        assert pool.has_key("alpha") is True
        assert pool.has_key("beta") is True

    def test_returns_false_for_missing_key(self):
        pool = KeyPool("svc", ["alpha", "beta"])
        assert pool.has_key("gamma") is False
        assert pool.has_key("") is False

    def test_returns_false_for_partial_match(self):
        pool = KeyPool("svc", ["px_abc123"])
        assert pool.has_key("px_abc") is False
        assert pool.has_key("px_abc1234") is False


# ──────────────────────────────────────────────
# AllKeysCooledDown exception
# ──────────────────────────────────────────────

class TestAllKeysCooledDownException:
    def test_message_contains_service(self):
        exc = AllKeysCooledDown("google_vision", 42.5)
        assert "google_vision" in str(exc)
        assert "42" in str(exc)

    def test_retry_after_attribute(self):
        exc = AllKeysCooledDown("svc", 30.0)
        assert exc.retry_after == 30.0
        assert exc.service == "svc"
