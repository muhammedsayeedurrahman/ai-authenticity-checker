"""
Pydantic request/response models for ProofyX REST API.

All responses follow the envelope pattern:
{success: bool, data: T | null, error: str | null}
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _to_float(v: Any) -> float:
    """Coerce numpy/torch scalars to a native Python float."""
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


class ProofyxBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    @field_validator("risk_score", mode="before", check_fields=False)
    @classmethod
    def clamp_risk_score(cls, v: Any) -> float:
        """Coerce to float and clamp to [0, 1]."""
        return max(0.0, min(1.0, _to_float(v)))

    @field_validator("risk_percent", mode="before", check_fields=False)
    @classmethod
    def clamp_risk_percent(cls, v: Any) -> float:
        """Coerce to float and clamp to [0, 100]."""
        return max(0.0, min(100.0, _to_float(v)))


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
    gradcam_overlay: Optional[str] = Field(
        default=None, description="Base64 PNG data URI, as emitted by the pipeline",
    )
    reverse_search: Optional[dict[str, Any]] = None


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
    fusion_mode: str = ""
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

class AudioAnalysisResult(ProofyxBase):
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
    flagged_modalities: list[str] = Field(default_factory=list)
    clean_modalities: list[str] = Field(default_factory=list)
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
# Document / ID Analysis
# ──────────────────────────────────────────────

class DocumentExifResponse(BaseModel):
    has_exif: bool = False
    suspicious: bool = False
    suspicion_score: float = 0.0
    findings: list[str] = Field(default_factory=list)
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None


class IdValidationResponse(BaseModel):
    valid: bool
    reason: str
    id_type: str
    id_label: str


class C2paResponse(BaseModel):
    valid: bool = False
    validation_state: Optional[str] = None
    generator: Optional[str] = None
    ai_generated_signal: bool = False
    actions: list[dict[str, Any]] = Field(default_factory=list)


class DocumentAnalysisResult(ProofyxBase):
    id: str = ""
    timestamp: str = ""
    risk_score: float = Field(ge=0, le=1)
    risk_percent: float = Field(ge=0, le=100)
    verdict: str
    confidence: str
    risk_level: str = ""
    primary_finding: str = ""
    ai_generated_likely: bool = False
    ai_generated_score: float = 0.0
    manipulation_likely: bool = False
    manipulation_score: float = 0.0
    checks: dict[str, str] = Field(default_factory=dict)
    ela_score: float = 0.0
    noise_consistency_score: float = 0.0
    copy_move_score: float = 0.0
    copy_move_matches: int = 0
    id_validation: Optional[IdValidationResponse] = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    exif: Optional[DocumentExifResponse] = None
    has_c2pa: bool = False
    c2pa: Optional[C2paResponse] = None
    reverse_search: Optional[dict[str, Any]] = None
    processing_time_ms: float = 0.0
    media_type: str = "document"


class DocumentAnalysisResponse(BaseModel):
    success: bool
    data: Optional[DocumentAnalysisResult] = None
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
    reverse_search_available: bool = False


class HealthResponse(BaseModel):
    status: str = "active"
    models_loaded: int = 0
    version: str = "2.0"
