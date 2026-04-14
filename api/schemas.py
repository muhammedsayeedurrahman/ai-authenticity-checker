"""
Pydantic request/response models for ProofyX REST API.

All responses follow the envelope pattern:
{success: bool, data: T | null, error: str | null}
"""

from __future__ import annotations

from datetime import datetime
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


class HealthResponse(BaseModel):
    status: str = "active"
    models_loaded: int = 0
    version: str = "2.0"
