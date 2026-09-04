"""Unit tests for core pipeline utilities."""

from __future__ import annotations


import pytest
from PIL import Image

from core.cybercrime_risk import (
    assess_audio_cybercrime_risk, assess_image_cybercrime_risk,
    assess_video_cybercrime_risk,
)
from core.pipeline import _empty_result, calibrate_score, forensic_score
from core.types import Confidence, RiskLevel, Verdict


# ──────────────────────────────────────────────
# calibrate_score
# ──────────────────────────────────────────────

class TestCalibrateScore:
    def test_midpoint_returns_midpoint(self):
        result = calibrate_score(0.5, temperature=1.0)
        assert abs(result - 0.5) < 1e-6

    def test_higher_temp_pulls_toward_center(self):
        raw = 0.9
        at_1 = calibrate_score(raw, temperature=1.0)
        at_2 = calibrate_score(raw, temperature=2.0)
        assert at_2 < at_1  # higher temp = less extreme

    def test_clamps_extreme_values(self):
        result_low = calibrate_score(0.0, temperature=1.2)
        result_high = calibrate_score(1.0, temperature=1.2)
        assert 0.0 < result_low < 0.5
        assert 0.5 < result_high < 1.0

    def test_output_in_valid_range(self):
        for score in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            result = calibrate_score(score)
            assert 0.0 <= result <= 1.0


# ──────────────────────────────────────────────
# forensic_score
# ──────────────────────────────────────────────

class TestForensicScore:
    def test_returns_float_in_range(self):
        img = Image.new("RGB", (128, 128), color=(200, 200, 200))
        score = forensic_score(img)
        assert 0.0 <= float(score) <= 1.0

    def test_uniform_image_scores_low(self):
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        score = forensic_score(img)
        assert score < 0.5  # uniform image = low manipulation signal

    def test_tiny_image_returns_zero(self):
        img = Image.new("RGB", (10, 10), color=(128, 128, 128))
        score = forensic_score(img)
        assert score == 0.0  # too small for patches


# ──────────────────────────────────────────────
# Verdict enum
# ──────────────────────────────────────────────

class TestVerdict:
    @pytest.mark.parametrize("risk,expected", [
        (0.0, Verdict.AUTHENTIC),
        (0.20, Verdict.AUTHENTIC),
        (0.59, Verdict.AUTHENTIC),
        (0.60, Verdict.AI_GENERATED),
        (0.75, Verdict.AI_GENERATED),
        (1.0, Verdict.AI_GENERATED),
    ])
    def test_from_risk_score(self, risk, expected):
        assert Verdict.from_risk_score(risk) == expected


class TestConfidence:
    @pytest.mark.parametrize("risk,expected", [
        (0.0, Confidence.HIGH),
        (0.5, Confidence.LOW),
        (0.35, Confidence.MEDIUM),
        (0.85, Confidence.HIGH),
    ])
    def test_from_risk_score(self, risk, expected):
        assert Confidence.from_risk_score(risk) == expected


class TestRiskLevel:
    @pytest.mark.parametrize("risk,expected", [
        (0.0, RiskLevel.MINIMAL),
        (0.30, RiskLevel.LOW),
        (0.50, RiskLevel.MEDIUM),
        (0.75, RiskLevel.HIGH),
        (0.90, RiskLevel.CRITICAL),
    ])
    def test_from_risk_score(self, risk, expected):
        assert RiskLevel.from_risk_score(risk) == expected


# ──────────────────────────────────────────────
# Cybercrime risk advisory
# ──────────────────────────────────────────────

class TestAssessAudioCybercrimeRisk:
    def test_below_threshold_returns_none_category(self):
        result = assess_audio_cybercrime_risk(risk_score=0.4)
        assert result["category"] == "none"
        assert result["label"] == ""
        assert result["signals"] == []

    def test_high_confidence_voice_clone_flags_fraud_pattern(self):
        result = assess_audio_cybercrime_risk(
            risk_score=0.85,
            manipulation_type="AI voice cloning / TTS",
            evidence=["unnatural harmonic structure", "no significant spectral artifacts"],
        )
        assert result["category"] == "voice_clone_fraud"
        assert "voice" in result["label"].lower()
        assert "85%" in result["signals"][0]
        # The "no artifacts" filler evidence string should not leak through.
        assert "no significant spectral artifacts" not in result["signals"]
        assert "unnatural harmonic structure" in result["signals"]
        assert result["disclaimer"]

    def test_at_threshold_flags(self):
        result = assess_audio_cybercrime_risk(risk_score=0.60)
        assert result["category"] == "voice_clone_fraud"


class TestAssessImageCybercrimeRisk:
    def test_requires_face_and_exif_suspicion_and_high_risk(self):
        base_kwargs = dict(risk_score=0.9, face_detected=True, exif_suspicious=True)

        assert assess_image_cybercrime_risk(**{**base_kwargs, "risk_score": 0.5})["category"] == "none"
        assert assess_image_cybercrime_risk(**{**base_kwargs, "face_detected": False})["category"] == "none"
        assert assess_image_cybercrime_risk(**{**base_kwargs, "exif_suspicious": False})["category"] == "none"

    def test_flags_synthetic_identity_pattern(self):
        result = assess_image_cybercrime_risk(
            risk_score=0.9,
            face_detected=True,
            exif_suspicious=True,
            exif_findings=["No EXIF metadata found (common in AI-generated images)"],
        )
        assert result["category"] == "synthetic_identity"
        assert "No EXIF metadata found (common in AI-generated images)" in result["signals"]
        assert result["advisory"]


class TestAssessVideoCybercrimeRisk:
    def test_requires_splice_signal(self):
        result = assess_video_cybercrime_risk(
            risk_score=0.9, faces_detected_in_frames=10,
            significant_jumps=0, score_variance=0.0,
        )
        assert result["category"] == "none"

    def test_flags_impersonation_pattern(self):
        result = assess_video_cybercrime_risk(
            risk_score=0.8, faces_detected_in_frames=12,
            significant_jumps=2, score_variance=0.03,
        )
        assert result["category"] == "impersonation_video"
        assert any("splice" in s for s in result["signals"])

    def test_no_faces_does_not_flag(self):
        result = assess_video_cybercrime_risk(
            risk_score=0.9, faces_detected_in_frames=0,
            significant_jumps=3, score_variance=0.05,
        )
        assert result["category"] == "none"


class TestEmptyResultComplianceLabel:
    def test_empty_result_carries_indeterminate_compliance_label(self):
        result = _empty_result("image", start_time=0.0, error="boom")
        assert "compliance_label" in result
        assert result["compliance_label"]["label_code"] == "indeterminate"
        assert result["compliance_label"]["sla_applies"] is False
