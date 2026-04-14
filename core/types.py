"""
Standardized types for ProofyX detection pipeline.

All models MUST return PredictionResult with probability = P(fake).
This permanently fixes the P(real)/P(fake) inversion bug.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class Verdict(str, Enum):
    """4-tier verdict system replacing HIGH/MEDIUM/LOW RISK."""
    LIKELY_MANIPULATED = "LIKELY MANIPULATED"
    POSSIBLY_MANIPULATED = "POSSIBLY MANIPULATED"
    UNCERTAIN = "UNCERTAIN"
    LIKELY_AUTHENTIC = "LIKELY AUTHENTIC"

    @classmethod
    def from_risk_score(cls, risk: float) -> "Verdict":
        if risk > 0.70:
            return cls.LIKELY_MANIPULATED
        if risk > 0.45:
            return cls.POSSIBLY_MANIPULATED
        if risk > 0.30:
            return cls.UNCERTAIN
        return cls.LIKELY_AUTHENTIC


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @classmethod
    def from_risk_score(cls, risk: float) -> "Confidence":
        distance = abs(risk - 0.5)
        if distance > 0.25:
            return cls.HIGH
        if distance > 0.10:
            return cls.MEDIUM
        return cls.LOW


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"

    @classmethod
    def from_risk_score(cls, risk: float) -> "RiskLevel":
        if risk > 0.85:
            return cls.CRITICAL
        if risk > 0.70:
            return cls.HIGH
        if risk > 0.45:
            return cls.MEDIUM
        if risk > 0.25:
            return cls.LOW
        return cls.MINIMAL


@dataclass(frozen=True)
class PredictionResult:
    """
    Standardized output from every model.

    probability is ALWAYS P(fake): 0.0 = certainly real, 1.0 = certainly fake.
    This contract is enforced by the base model class.
    """
    model: str
    probability: float
    prediction: int  # 1 = fake, 0 = real
    class_name: str  # "fake" or "real"
    inference_time: float

    @classmethod
    def create(cls, model: str, probability: float,
               threshold: float = 0.5, inference_time: float = 0.0) -> "PredictionResult":
        prediction = 1 if probability >= threshold else 0
        return cls(
            model=model,
            probability=max(0.0, min(1.0, probability)),
            prediction=prediction,
            class_name="fake" if prediction == 1 else "real",
            inference_time=inference_time,
        )


@dataclass
class ModelScore:
    """Per-model score for reports and API responses."""
    name: str
    probability: float  # P(fake) 0.0-1.0
    verdict: str
    confidence: float  # 0-100
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    heatmap_base64: Optional[str] = None


@dataclass
class ExifMetadata:
    """Extracted EXIF metadata from image files."""
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    timestamp: Optional[str] = None
    software: Optional[str] = None
    gps_coordinates: Optional[str] = None
    orientation: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    has_exif: bool = False
    suspicious: bool = False
    suspicion_score: float = 0.0
    findings: list[str] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class TemporalAnalysis:
    """Video temporal analysis results."""
    score_variance: float = 0.0
    max_frame_jump: float = 0.0
    significant_jumps: int = 0
    risk_timeline: list[float] = field(default_factory=list)


@dataclass
class AudioResult:
    """Audio analysis results."""
    risk_score: float = 0.0
    authenticity_score: float = 100.0
    verdict: str = ""
    confidence: str = ""
    manipulation_type: str = ""
    evidence: list[str] = field(default_factory=list)
    duration_sec: float = 0.0
    segments_analyzed: int = 0
    processing_time_ms: float = 0.0


@dataclass
class AnalysisResult:
    """Complete analysis result for any media type."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    media_type: str = "image"

    # Overall scores
    risk_score: float = 0.0
    risk_percent: float = 0.0
    verdict: str = ""
    confidence: str = ""
    risk_level: str = ""

    # Model details
    model_scores: dict[str, float] = field(default_factory=dict)
    model_agreement: str = ""
    fusion_mode: str = ""
    models_used: int = 0

    # Image-specific
    face_detected: bool = False
    face_aligned: bool = False
    gradcam_image: Any = None
    original_image: Any = None

    # Video-specific
    total_frames_analyzed: int = 0
    fake_frames: int = 0
    real_frames: int = 0
    temporal_analysis: Optional[TemporalAnalysis] = None
    frame_results: list[dict] = field(default_factory=list)

    # Audio-specific
    audio_result: Optional[AudioResult] = None

    # Multimodal
    modality_scores: dict[str, Optional[float]] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    exif: Optional[ExifMetadata] = None
    processing_time_ms: float = 0.0
    explanation: str = ""

    # For reports
    file_name: str = ""


@dataclass
class ForensicReport:
    """
    Complete forensic report for PDF/HTML generation.
    Adapted from DeepGuard's report model, extended for multimodal.
    """
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    media_type: str = "image"
    file_name: str = ""

    # Overall verdict
    overall_verdict: str = ""
    overall_confidence: float = 0.0
    risk_level: str = ""
    risk_score: float = 0.0

    # Per-model results
    model_results: dict[str, ModelScore] = field(default_factory=dict)

    # Modality-specific
    audio_result: Optional[AudioResult] = None
    temporal_analysis: Optional[TemporalAnalysis] = None
    total_frames_analyzed: int = 0
    fake_frames: int = 0

    # Visuals (base64)
    source_image_base64: Optional[str] = None
    gradcam_overlay_base64: Optional[str] = None

    # Metadata
    exif_metadata: Optional[ExifMetadata] = None
    image_dimensions: Optional[tuple[int, int]] = None
    processing_time_ms: float = 0.0
    fusion_mode: str = ""
    models_used: int = 0
    face_detected: bool = False

    # Forensic findings
    explanation: str = ""
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
