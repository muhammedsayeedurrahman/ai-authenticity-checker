"""Tests for core/audit_hash.py — pure hash-chain helpers."""

from __future__ import annotations

from core.audit_hash import canonical_json, entry_hash


class TestCanonicalJson:
    def test_sorted_keys_produce_identical_output_regardless_of_input_order(self):
        a = canonical_json({"b": 1, "a": 2})
        b = canonical_json({"a": 2, "b": 1})
        assert a == b

    def test_no_extraneous_whitespace(self):
        assert " " not in canonical_json({"a": 1, "b": [1, 2]})


class TestEntryHash:
    def test_deterministic_for_same_inputs(self):
        h1 = entry_hash("prev", "2026-01-01T00:00:00+00:00", "content.labeled", "sub-1", {"x": 1})
        h2 = entry_hash("prev", "2026-01-01T00:00:00+00:00", "content.labeled", "sub-1", {"x": 1})
        assert h1 == h2

    def test_different_payload_changes_hash(self):
        h1 = entry_hash("prev", "t", "event", "sub", {"x": 1})
        h2 = entry_hash("prev", "t", "event", "sub", {"x": 2})
        assert h1 != h2

    def test_different_prev_hash_changes_hash(self):
        h1 = entry_hash("prev-a", "t", "event", "sub", {})
        h2 = entry_hash("prev-b", "t", "event", "sub", {})
        assert h1 != h2

    def test_none_prev_hash_normalizes_same_as_empty_string(self):
        h1 = entry_hash(None, "t", "event", "sub", {})
        h2 = entry_hash("", "t", "event", "sub", {})
        assert h1 == h2

    def test_returns_hex_sha256_digest(self):
        h = entry_hash(None, "t", "event", "sub", {})
        assert len(h) == 64
        int(h, 16)  # must not raise

    def test_different_actor_id_changes_hash(self):
        """Actor attribution must be tamper-evident too — a DB-admin edit
        reassigning who performed an action must break the chain."""
        h1 = entry_hash(None, "t", "event", "sub", {}, actor_type="user", actor_id="employee-1")
        h2 = entry_hash(None, "t", "event", "sub", {}, actor_type="user", actor_id="employee-2")
        assert h1 != h2

    def test_different_actor_type_changes_hash(self):
        h1 = entry_hash(None, "t", "event", "sub", {}, actor_type="user", actor_id="x")
        h2 = entry_hash(None, "t", "event", "sub", {}, actor_type="system", actor_id="x")
        assert h1 != h2

    def test_different_subject_type_changes_hash(self):
        h1 = entry_hash(None, "t", "event", "sub", {}, subject_type="sla_clock")
        h2 = entry_hash(None, "t", "event", "sub", {}, subject_type="content_label")
        assert h1 != h2

    def test_default_actor_fields_match_omitted_call(self):
        """Backward compatibility: a call that omits the new params hashes
        identically to one that passes empty-string defaults explicitly."""
        h1 = entry_hash(None, "t", "event", "sub", {})
        h2 = entry_hash(None, "t", "event", "sub", {}, actor_type="", actor_id="", subject_type="")
        assert h1 == h2
