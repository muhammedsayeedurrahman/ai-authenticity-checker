"""Unit tests for core pipeline utilities."""

from __future__ import annotations

import math

import pytest
from PIL import Image

from core.pipeline import calibrate_score, forensic_score
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
        (0.0, Verdict.LIKELY_AUTHENTIC),
        (0.20, Verdict.LIKELY_AUTHENTIC),
        (0.35, Verdict.UNCERTAIN),
        (0.50, Verdict.POSSIBLY_MANIPULATED),
        (0.75, Verdict.LIKELY_MANIPULATED),
        (1.0, Verdict.LIKELY_MANIPULATED),
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
