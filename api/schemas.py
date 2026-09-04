"""
Pydantic request/response models for ProofyX REST API.

All responses follow the envelope pattern:
{success: bool, data: T | null, error: str | null}
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProofyxBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# ──────────────────────────────────────────────
# Shared sub-models
# ──────────────────────────────────────────────

class ModelScoreResponse(BaseModel):
    name: str
    score: float = Field(ge=0, le=1, description="P(fake) 0.0-1.0")
    confidence: float = Field(ge=0, le=100, description="Confidence 0-100")


class TemporalAnalysisResponse(BaseModel):
    score_variance: float = 0.0
    max_frame_jump: float = 0.0
    significant_jumps: int = 0
    risk_timeline: list[float] = Field(default_factory=list)


class AudioResultResponse(BaseModel):
    risk_score: float = 0.0
    authenticity_score: float = 100.0
    verdict: str = ""
    confidence: str = ""
    manipulation_type: str = ""
    evidence: list[str] = Field(default_factory=list)
    duration_sec: float = 0.0
    segments_analyzed: int = 0


class ExifResponse(BaseModel):
    has_exif: bool = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    timestamp: Optional[str] = None
    software: Optional[str] = None
    gps_coordinates: Optional[str] = None
    suspicious: bool = False
    suspicion_score: float = 0.0
    findings: list[str] = Field(default_factory=list)


class CybercrimeRiskResponse(BaseModel):
    """Plain-language advisory for deepfake-enabled fraud patterns.

    category == "none" (the default) means no fraud pattern was flagged;
    label/description/advisory/signals are only populated otherwise.
    See core/cybercrime_risk.py for the thresholds and category definitions.
    """
    category: str = "none"
    label: str = ""
    description: str = ""
    advisory: str = ""
    signals: list[str] = Field(default_factory=list)
    disclaimer: str = ""


class ComplianceLabelResponse(BaseModel):
    """India IT Rules 2026 labeling/traceability determination for one analysis.

    Heuristic compliance-workflow aid, not legal advice — see
    core/compliance_label.py. Every field has a default so an absent
    label (e.g. an older stored analysis) never fails validation.
    """
    label_code: str = "indeterminate"
    label_display: str = ""
    requires_visible_label: bool = False
    requires_embedded_metadata: bool = False
    label_basis: list[str] = Field(default_factory=list)
    regulatory_basis: str = ""
    ruleset_version: str = ""
    detector_version: str = ""
    risk_score: float = 0.0
    confidence: str = ""
    recommended_action: str = "none"
    sla_applies: bool = False
    sla_deadline_seconds: Optional[int] = None
    assessed_at: str = ""
    disclaimer: str = ""


# ──────────────────────────────────────────────
# Image Analysis
# ──────────────────────────────────────────────

class ImageAnalysisResult(ProofyxBase):
    id: str = ""
    timestamp: str = ""
    risk_score: float = Field(ge=0, le=1)
    risk_percent: float = Field(ge=0, le=100)
    verdict: str
    confidence: str
    risk_level: str = ""
    model_agreement: str = ""
    model_scores: dict[str, float] = Field(default_factory=dict)
    fusion_mode: str = ""
    face_detected: bool = False
    face_aligned: bool = False
    models_used: int = 0
    processing_time_ms: float = 0.0
    media_type: str = "image"
    explanation: str = ""
    metadata: Optional[dict[str, Any]] = None
    cybercrime_risk: Optional[CybercrimeRiskResponse] = None
    compliance_label: Optional[ComplianceLabelResponse] = None
    gradcam_image: Optional[str] = Field(
        default=None, description="Bare base64 PNG (no data-URI prefix)",
    )


class ImageAnalysisResponse(BaseModel):
    success: bool
    data: Optional[ImageAnalysisResult] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Video Analysis
# ──────────────────────────────────────────────

class FrameResult(ProofyxBase):
    frame_index: int
    timestamp: float
    risk_score: float
    has_face: bool = False
    model_scores: dict[str, float] = Field(default_factory=dict)


class VideoAnalysisResult(ProofyxBase):
    id: str = ""
    timestamp: str = ""
    risk_score: float = Field(ge=0, le=1)
    risk_percent: float = Field(ge=0, le=100)
    verdict: str
    confidence: str
    prediction: str = ""
    total_frames_analyzed: int = 0
    fake_frames: int = 0
    real_frames: int = 0
    temporal_analysis: Optional[TemporalAnalysisResponse] = None
    video_info: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = 0.0
    media_type: str = "video"
    cybercrime_risk: Optional[CybercrimeRiskResponse] = None
    compliance_label: Optional[ComplianceLabelResponse] = None


class VideoAnalysisResponse(BaseModel):
    success: bool
    data: Optional[VideoAnalysisResult] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Audio Analysis
# ──────────────────────────────────────────────

class AudioAnalysisResult(BaseModel):
    id: str = ""
    timestamp: str = ""
    risk_score: float = Field(ge=0, le=1)
    authenticity_score: float = Field(ge=0, le=100)
    verdict: str
    confidence: str
    manipulation_type: str = ""
    evidence: list[str] = Field(default_factory=list)
    duration_sec: float = 0.0
    segments_analyzed: int = 0
    processing_time_ms: float = 0.0
    media_type: str = "audio"
    explanation: str = ""
    cybercrime_risk: Optional[CybercrimeRiskResponse] = None
    compliance_label: Optional[ComplianceLabelResponse] = None


class AudioAnalysisResponse(BaseModel):
    success: bool
    data: Optional[AudioAnalysisResult] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Multimodal Analysis
# ──────────────────────────────────────────────

class MultimodalAnalysisResult(ProofyxBase):
    id: str = ""
    timestamp: str = ""
    risk_score: float = Field(ge=0, le=1)
    risk_percent: float = Field(ge=0, le=100)
    verdict: str
    confidence: str
    media_types: list[str] = Field(default_factory=list)
    modality_scores: dict[str, Optional[float]] = Field(default_factory=dict)
    fusion_weights: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""
    processing_time_ms: float = 0.0
    media_type: str = "multimodal"
    cybercrime_risks: list[CybercrimeRiskResponse] = Field(default_factory=list)
    compliance_label: Optional[ComplianceLabelResponse] = None


class MultimodalAnalysisResponse(BaseModel):
    success: bool
    data: Optional[MultimodalAnalysisResult] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────
# History
# ──────────────────────────────────────────────

class HistoryEntry(BaseModel):
    id: str
    timestamp: str
    media_type: str
    risk_score: float
    verdict: str
    confidence: str
    models_used: int = 0
    processing_time_ms: float = 0.0
    file_name: str = ""
    user_id: Optional[str] = None


class HistoryListResponse(BaseModel):
    success: bool
    data: list[HistoryEntry] = Field(default_factory=list)
    total: int = 0


# ──────────────────────────────────────────────
# System
# ──────────────────────────────────────────────

class ModelStatus(BaseModel):
    loaded: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    total: int = 0
    corefakenet_ready: bool = False
    device: str = "cpu"


class HealthResponse(BaseModel):
    status: str = "active"
    models_loaded: int = 0
    version: str = "2.0"
