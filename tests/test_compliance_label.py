"""Tests for core/compliance_label.py — India IT Rules 2026 labeling layer."""

from __future__ import annotations

import json

from core.compliance_label import (
    DEFAULT_SLA_SECONDS,
    LABEL_RULESET_VERSION,
    LabelCode,
    build_compliance_label,
)


class TestTaxonomyBoundaries:
    def test_high_risk_high_confidence_is_synthetically_generated(self):
        label = build_compliance_label(risk_score=0.90, confidence="HIGH")
        assert label["label_code"] == LabelCode.SYNTHETICALLY_GENERATED.value

    def test_high_risk_low_confidence_is_possibly_synthetic(self):
        label = build_compliance_label(risk_score=0.62, confidence="LOW")
        assert label["label_code"] == LabelCode.POSSIBLY_SYNTHETIC.value

    def test_high_risk_medium_confidence_is_possibly_synthetic(self):
        label = build_compliance_label(risk_score=0.75, confidence="MEDIUM")
        assert label["label_code"] == LabelCode.POSSIBLY_SYNTHETIC.value

    def test_exactly_at_flagged_threshold_with_high_confidence(self):
        label = build_compliance_label(risk_score=0.60, confidence="HIGH")
        assert label["label_code"] == LabelCode.SYNTHETICALLY_GENERATED.value

    def test_mid_band_is_possibly_synthetic(self):
        label = build_compliance_label(risk_score=0.50, confidence="LOW")
        assert label["label_code"] == LabelCode.POSSIBLY_SYNTHETIC.value

    def test_exactly_at_advisory_threshold_is_possibly_synthetic(self):
        label = build_compliance_label(risk_score=0.45, confidence="LOW")
        assert label["label_code"] == LabelCode.POSSIBLY_SYNTHETIC.value

    def test_just_below_advisory_threshold_is_clean(self):
        label = build_compliance_label(risk_score=0.44, confidence="LOW")
        assert label["label_code"] == LabelCode.NO_SYNTHETIC_INDICATORS.value

    def test_zero_risk_is_clean(self):
        label = build_compliance_label(risk_score=0.0, confidence="HIGH")
        assert label["label_code"] == LabelCode.NO_SYNTHETIC_INDICATORS.value

    def test_error_forces_indeterminate_regardless_of_risk_score(self):
        label = build_compliance_label(risk_score=0.95, confidence="HIGH", error=True)
        assert label["label_code"] == LabelCode.INDETERMINATE.value


class TestVisibleLabelAndMetadataFlags:
    def test_synthetically_generated_requires_visible_label_and_embedded_metadata(self):
        label = build_compliance_label(risk_score=0.90, confidence="HIGH")
        assert label["requires_visible_label"] is True
        assert label["requires_embedded_metadata"] is True

    def test_possibly_synthetic_requires_visible_label_but_not_metadata(self):
        label = build_compliance_label(risk_score=0.50, confidence="LOW")
        assert label["requires_visible_label"] is True
        assert label["requires_embedded_metadata"] is False

    def test_clean_requires_neither(self):
        label = build_compliance_label(risk_score=0.10, confidence="HIGH")
        assert label["requires_visible_label"] is False
        assert label["requires_embedded_metadata"] is False


class TestSlaApplies:
    def test_flagged_with_cybercrime_category_starts_sla(self):
        label = build_compliance_label(
            risk_score=0.90, confidence="HIGH", cybercrime_category="voice_clone_fraud",
        )
        assert label["sla_applies"] is True
        assert label["sla_deadline_seconds"] == DEFAULT_SLA_SECONDS

    def test_flagged_with_explicit_complaint_starts_sla_even_without_category(self):
        label = build_compliance_label(
            risk_score=0.90, confidence="HIGH", cybercrime_category="none",
            flagged_by_complaint=True,
        )
        assert label["sla_applies"] is True

    def test_flagged_without_category_or_complaint_does_not_start_sla(self):
        label = build_compliance_label(
            risk_score=0.90, confidence="HIGH", cybercrime_category="none",
        )
        assert label["sla_applies"] is False
        assert label["sla_deadline_seconds"] is None

    def test_possibly_synthetic_never_starts_sla_even_with_category(self):
        label = build_compliance_label(
            risk_score=0.50, confidence="LOW", cybercrime_category="voice_clone_fraud",
        )
        assert label["sla_applies"] is False

    def test_error_never_starts_sla(self):
        label = build_compliance_label(
            risk_score=0.99, confidence="HIGH", cybercrime_category="voice_clone_fraud",
            error=True,
        )
        assert label["sla_applies"] is False


class TestLabelBasisAndMetadata:
    def test_label_basis_defaults_to_empty_list(self):
        label = build_compliance_label(risk_score=0.90, confidence="HIGH")
        assert label["label_basis"] == []

    def test_label_basis_passthrough(self):
        basis = ["Face manipulation detected", "3 abrupt score jumps"]
        label = build_compliance_label(risk_score=0.90, confidence="HIGH", label_basis=basis)
        assert label["label_basis"] == basis

    def test_ruleset_and_detector_version_present(self):
        label = build_compliance_label(
            risk_score=0.90, confidence="HIGH", detector_version="proofyx/learned",
        )
        assert label["ruleset_version"] == LABEL_RULESET_VERSION
        assert label["detector_version"] == "proofyx/learned"

    def test_disclaimer_always_present(self):
        for risk in (0.0, 0.5, 0.9):
            label = build_compliance_label(risk_score=risk, confidence="HIGH")
            assert label["disclaimer"]
            assert "not legal advice" in label["disclaimer"]

    def test_regulatory_basis_present(self):
        label = build_compliance_label(risk_score=0.90, confidence="HIGH")
        assert "India" in label["regulatory_basis"]

    def test_assessed_at_present(self):
        label = build_compliance_label(risk_score=0.90, confidence="HIGH")
        assert label["assessed_at"]

    def test_output_is_json_serializable(self):
        label = build_compliance_label(
            risk_score=0.90, confidence="HIGH", label_basis=["x"],
            cybercrime_category="voice_clone_fraud",
        )
        json.dumps(label)  # must not raise

    def test_two_calls_produce_independent_dicts(self):
        a = build_compliance_label(risk_score=0.90, confidence="HIGH", label_basis=["a"])
        b = build_compliance_label(risk_score=0.10, confidence="LOW")
        a["label_basis"].append("mutated")
        assert b["label_basis"] == []
