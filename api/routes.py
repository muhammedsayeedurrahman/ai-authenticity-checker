"""
FastAPI REST API endpoints for ProofyX.

All endpoints share the same core pipeline as the Gradio UI.
Responses follow the envelope pattern: {success, data, error}.
"""

from __future__ import annotations

import io
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Header, UploadFile, HTTPException, Query
from PIL import Image

from core.secrets import get_pool
from api.schemas import (
    ImageAnalysisResponse, ImageAnalysisResult,
    VideoAnalysisResponse, VideoAnalysisResult,
    AudioAnalysisResponse, AudioAnalysisResult,
    MultimodalAnalysisResponse, MultimodalAnalysisResult,
    HistoryListResponse, HistoryEntry,
    ModelStatus, HealthResponse,
)
from core.pipeline import (
    analyze_image, analyze_video, analyze_audio,
    analyze_multimodal, get_registry,
)
from db.history import AnalysisHistory

router = APIRouter()
history = AnalysisHistory()


# ──────────────────────────────────────────────
# API Key Authentication
# ──────────────────────────────────────────────

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """Validate the X-API-Key header against the PROOFYX_API_KEY pool.

    - If no keys are configured → dev mode, all requests pass through.
    - If keys are configured → the header must match one of them.
    """
    pool = get_pool("PROOFYX_API_KEY")

    if pool is None:
        # No keys configured — dev mode (unauthenticated)
        return None

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    if not pool.has_key(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")

    return x_api_key


# ──────────────────────────────────────────────
# Analysis Endpoints
# ──────────────────────────────────────────────

@router.post("/analyze/image", response_model=ImageAnalysisResponse)
async def api_analyze_image(
    file: UploadFile = File(...),
    mode: str = Query("ensemble", regex="^(ensemble|fast)$"),
    _key: Optional[str] = Depends(verify_api_key),
):
    """Analyze an uploaded image for deepfake indicators."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = analyze_image(image, mode=mode)

    if "error" in result and result["error"]:
        return ImageAnalysisResponse(success=False, error=result["error"])

    analysis_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    # Persist to history
    result["id"] = analysis_id
    result["timestamp"] = timestamp
    result["file_name"] = file.filename or ""
    history.save(result)

    # Remove non-serializable fields
    result.pop("gradcam_image", None)
    result.pop("original_image", None)

    return ImageAnalysisResponse(
        success=True,
        data=ImageAnalysisResult(id=analysis_id, timestamp=timestamp, **{
            k: v for k, v in result.items()
            if k in ImageAnalysisResult.model_fields
        }),
    )


@router.post("/analyze/video", response_model=VideoAnalysisResponse)
async def api_analyze_video(
    file: UploadFile = File(...),
    fps: float = Query(4.0, ge=0.5, le=30),
    aggregation: str = Query("weighted_avg"),
    _key: Optional[str] = Depends(verify_api_key),
):
    """Analyze an uploaded video for deepfake indicators."""
    try:
        contents = await file.read()
        suffix = ".mp4"
        if file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid video file")

    result = analyze_video(tmp_path, fps=fps, aggregation=aggregation)

    if "error" in result and result["error"]:
        return VideoAnalysisResponse(success=False, error=result["error"])

    analysis_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    result["id"] = analysis_id
    result["timestamp"] = timestamp
    result["file_name"] = file.filename or ""
    history.save(result)

    return VideoAnalysisResponse(
        success=True,
        data=VideoAnalysisResult(id=analysis_id, timestamp=timestamp, **{
            k: v for k, v in result.items()
            if k in VideoAnalysisResult.model_fields
        }),
    )


@router.post("/analyze/audio", response_model=AudioAnalysisResponse)
async def api_analyze_audio(
    file: UploadFile = File(...),
    _key: Optional[str] = Depends(verify_api_key),
):
    """Analyze an uploaded audio file for deepfake indicators."""
    try:
        contents = await file.read()
        suffix = ".wav"
        if file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio file")

    result = analyze_audio(tmp_path)

    if "error" in result and result["error"]:
        return AudioAnalysisResponse(success=False, error=result["error"])

    analysis_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    result["id"] = analysis_id
    result["timestamp"] = timestamp
    result["file_name"] = file.filename or ""
    history.save(result)

    return AudioAnalysisResponse(
        success=True,
        data=AudioAnalysisResult(id=analysis_id, timestamp=timestamp, **{
            k: v for k, v in result.items()
            if k in AudioAnalysisResult.model_fields
        }),
    )


@router.post("/analyze/multimodal", response_model=MultimodalAnalysisResponse)
async def api_analyze_multimodal(
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    _key: Optional[str] = Depends(verify_api_key),
):
    """Analyze multiple media types with cross-modal fusion."""
    image_pil = None
    video_path = None
    audio_path = None

    if image is not None:
        try:
            contents = await image.read()
            image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            pass

    if video is not None:
        try:
            contents = await video.read()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(contents)
                video_path = tmp.name
        except Exception:
            pass

    if audio is not None:
        try:
            contents = await audio.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(contents)
                audio_path = tmp.name
        except Exception:
            pass

    if image_pil is None and video_path is None and audio_path is None:
        raise HTTPException(status_code=400, detail="No valid media files provided")

    result = analyze_multimodal(image=image_pil, video_path=video_path, audio_path=audio_path)

    if "error" in result and result["error"]:
        return MultimodalAnalysisResponse(success=False, error=result["error"])

    analysis_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    return MultimodalAnalysisResponse(
        success=True,
        data=MultimodalAnalysisResult(id=analysis_id, timestamp=timestamp, **{
            k: v for k, v in result.items()
            if k in MultimodalAnalysisResult.model_fields
        }),
    )


# ──────────────────────────────────────────────
# History Endpoints
# ──────────────────────────────────────────────

@router.get("/history", response_model=HistoryListResponse)
async def list_history(
    limit: int = Query(20, ge=1, le=100),
    media_type: Optional[str] = Query(None),
):
    """List recent analyses."""
    rows = history.get_recent(limit=limit, media_type=media_type)
    entries = [
        HistoryEntry(**{k: v for k, v in row.items() if k in HistoryEntry.model_fields})
        for row in rows
    ]
    return HistoryListResponse(success=True, data=entries, total=history.count())


@router.get("/history/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get a specific analysis result."""
    result = history.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"success": True, "data": result}


# ──────────────────────────────────────────────
# System Endpoints
# ──────────────────────────────────────────────

@router.get("/models/status", response_model=ModelStatus)
async def models_status():
    """List loaded models and their status."""
    reg = get_registry()
    return ModelStatus(**reg.get_status())


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    reg = get_registry()
    return HealthResponse(
        status="active",
        models_loaded=len(reg.loaded),
    )
