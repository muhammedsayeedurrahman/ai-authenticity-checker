"""
FastAPI REST API endpoints for ProofyX.

All endpoints share the same core pipeline as the Gradio UI.
Responses follow the envelope pattern: {success, data, error}.
"""

import asyncio
import io
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import (
    APIRouter, Depends, File, HTTPException, Query, Request, UploadFile,
)
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.schemas import (
    AudioAnalysisResponse, AudioAnalysisResult,
    HealthResponse,
    HistoryEntry, HistoryListResponse,
    ImageAnalysisResponse, ImageAnalysisResult,
    ModelStatus,
    MultimodalAnalysisResponse, MultimodalAnalysisResult,
    VideoAnalysisResponse, VideoAnalysisResult,
)
from core.pipeline import (
    analyze_audio, analyze_image, analyze_multimodal,
    analyze_video, get_registry,
)
from core.auth import get_current_user
from db.history import AnalysisHistory

logger = logging.getLogger(__name__)

router = APIRouter()
history = AnalysisHistory()
limiter = Limiter(key_func=get_remote_address)

# Serialize GPU inference — prevents concurrent model.forward() calls from
# corrupting CUDA state or producing wrong results.  Set > 1 only if models
# are isolated per-worker or you have verified thread-safe inference.
_MAX_CONCURRENT_INFERENCE = int(os.environ.get("PROOFYX_MAX_CONCURRENT", "1"))
_inference_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_INFERENCE)

# ──────────────────────────────────────────────
# Upload Validation Constants
# ──────────────────────────────────────────────

MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB
MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100 MB

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Per-media-type analysis timeouts (seconds), configurable via env vars
TIMEOUT_IMAGE = int(os.environ.get("PROOFYX_TIMEOUT_IMAGE", "60"))
TIMEOUT_VIDEO = int(os.environ.get("PROOFYX_TIMEOUT_VIDEO", "600"))
TIMEOUT_AUDIO = int(os.environ.get("PROOFYX_TIMEOUT_AUDIO", "90"))
TIMEOUT_MULTIMODAL = int(os.environ.get("PROOFYX_TIMEOUT_MULTIMODAL", "300"))

# Magic bytes for image format validation
MAGIC_BYTES: dict[str, list[bytes]] = {
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png": [b"\x89PNG"],
    ".webp": [b"RIFF"],
    ".bmp": [b"BM"],
    ".tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
}


def _validate_magic_bytes(contents: bytes, ext: str) -> None:
    """Validate file contents match expected magic bytes for the extension."""
    expected = MAGIC_BYTES.get(ext)
    if expected is None:
        return
    for magic in expected:
        if contents[:len(magic)] == magic:
            return
    raise HTTPException(
        status_code=400,
        detail=f"File content does not match expected format for {ext}",
    )


async def _read_validated(
    file: UploadFile, max_size: int, allowed_ext: set[str],
) -> bytes:
    """Read and validate an uploaded file (size + extension + magic bytes for images)."""
    ext = ""
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext and ext not in allowed_ext:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    contents = await file.read()
    if len(contents) > max_size:
        max_mb = max_size // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large. Maximum: {max_mb}MB")

    # Validate magic bytes for image uploads (video/audio containers are too complex)
    if ext in ALLOWED_IMAGE_EXT:
        _validate_magic_bytes(contents, ext)

    return contents


def _safe_tmp_remove(path: Optional[str]) -> None:
    """Silently remove a temp file if it exists."""
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


async def _run_with_timeout(
    fn: Callable[..., Any],
    timeout: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a sync function in a thread pool with a timeout.

    Acquires the inference semaphore first to prevent concurrent GPU access,
    then runs *fn* in a worker thread with *timeout* seconds deadline.
    Raises HTTP 504 on timeout, HTTP 503 if the semaphore cannot be acquired
    within 5 seconds (server overloaded).
    """
    try:
        acquired = await asyncio.wait_for(
            _inference_semaphore.acquire(), timeout=5.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server busy — too many concurrent analysis requests",
        )

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Analysis timed out after %ds: %s", timeout, fn.__name__,
        )
        raise HTTPException(
            status_code=504,
            detail=f"Analysis timed out after {timeout}s",
        )
    finally:
        _inference_semaphore.release()


# ──────────────────────────────────────────────
# Analysis Endpoints
# ──────────────────────────────────────────────

@router.post("/analyze/image", response_model=ImageAnalysisResponse)
@limiter.limit("30/minute")
async def api_analyze_image(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Query("ensemble", pattern="^(ensemble|fast)$"),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Analyze an uploaded image for deepfake indicators."""
    contents = await _read_validated(file, MAX_IMAGE_SIZE, ALLOWED_IMAGE_EXT)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = await _run_with_timeout(analyze_image, TIMEOUT_IMAGE, image, mode=mode)

    if "error" in result and result["error"]:
        return ImageAnalysisResponse(success=False, error=result["error"])

    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    user_id = current_user["id"] if current_user else None

    result["id"] = analysis_id
    result["timestamp"] = timestamp
    result["file_name"] = file.filename or ""
    await history.save(result, user_id=user_id)

    result.pop("gradcam_image", None)
    result.pop("original_image", None)

    return ImageAnalysisResponse(
        success=True,
        data=ImageAnalysisResult(id=analysis_id, timestamp=timestamp, **{
            k: v for k, v in result.items()
            if k in ImageAnalysisResult.model_fields and k not in ("id", "timestamp")
        }),
    )


@router.post("/analyze/video", response_model=VideoAnalysisResponse)
@limiter.limit("10/minute")
async def api_analyze_video(
    request: Request,
    file: UploadFile = File(...),
    fps: float = Query(4.0, ge=0.5, le=30),
    aggregation: str = Query("weighted_avg"),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Analyze an uploaded video for deepfake indicators."""
    contents = await _read_validated(file, MAX_VIDEO_SIZE, ALLOWED_VIDEO_EXT)

    suffix = ".mp4"
    if file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = await _run_with_timeout(
            analyze_video, TIMEOUT_VIDEO, tmp_path, fps=fps, aggregation=aggregation,
        )
    finally:
        _safe_tmp_remove(tmp_path)

    if "error" in result and result["error"]:
        return VideoAnalysisResponse(success=False, error=result["error"])

    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    user_id = current_user["id"] if current_user else None
    result["id"] = analysis_id
    result["timestamp"] = timestamp
    result["file_name"] = file.filename or ""
    await history.save(result, user_id=user_id)

    return VideoAnalysisResponse(
        success=True,
        data=VideoAnalysisResult(id=analysis_id, timestamp=timestamp, **{
            k: v for k, v in result.items()
            if k in VideoAnalysisResult.model_fields and k not in ("id", "timestamp")
        }),
    )


@router.post("/analyze/audio", response_model=AudioAnalysisResponse)
@limiter.limit("20/minute")
async def api_analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Analyze an uploaded audio file for deepfake indicators."""
    contents = await _read_validated(file, MAX_AUDIO_SIZE, ALLOWED_AUDIO_EXT)

    suffix = ".wav"
    if file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = await _run_with_timeout(analyze_audio, TIMEOUT_AUDIO, tmp_path)
    finally:
        _safe_tmp_remove(tmp_path)

    if "error" in result and result["error"]:
        return AudioAnalysisResponse(success=False, error=result["error"])

    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    user_id = current_user["id"] if current_user else None
    result["id"] = analysis_id
    result["timestamp"] = timestamp
    result["file_name"] = file.filename or ""
    await history.save(result, user_id=user_id)

    return AudioAnalysisResponse(
        success=True,
        data=AudioAnalysisResult(id=analysis_id, timestamp=timestamp, **{
            k: v for k, v in result.items()
            if k in AudioAnalysisResult.model_fields and k not in ("id", "timestamp")
        }),
    )


@router.post("/analyze/multimodal", response_model=MultimodalAnalysisResponse)
@limiter.limit("10/minute")
async def api_analyze_multimodal(
    request: Request,
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Analyze multiple media types with cross-modal fusion."""
    image_pil = None
    video_path = None
    audio_path = None

    try:
        if image is not None:
            contents = await _read_validated(image, MAX_IMAGE_SIZE, ALLOWED_IMAGE_EXT)
            try:
                image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            except (OSError, ValueError, Image.DecompressionBombError):
                pass

        if video is not None:
            contents = await _read_validated(video, MAX_VIDEO_SIZE, ALLOWED_VIDEO_EXT)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(contents)
                video_path = tmp.name

        if audio is not None:
            contents = await _read_validated(audio, MAX_AUDIO_SIZE, ALLOWED_AUDIO_EXT)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(contents)
                audio_path = tmp.name

        if image_pil is None and video_path is None and audio_path is None:
            raise HTTPException(status_code=400, detail="No valid media files provided")

        result = await _run_with_timeout(
            analyze_multimodal, TIMEOUT_MULTIMODAL,
            image=image_pil, video_path=video_path, audio_path=audio_path,
        )
    finally:
        _safe_tmp_remove(video_path)
        _safe_tmp_remove(audio_path)

    if "error" in result and result["error"]:
        return MultimodalAnalysisResponse(success=False, error=result["error"])

    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    user_id = current_user["id"] if current_user else None

    # Determine file_name from first available upload
    file_name = ""
    for upload in (image, video, audio):
        if upload is not None and upload.filename:
            file_name = upload.filename
            break

    result["id"] = analysis_id
    result["timestamp"] = timestamp
    result["file_name"] = file_name
    await history.save(result, user_id=user_id)

    return MultimodalAnalysisResponse(
        success=True,
        data=MultimodalAnalysisResult(id=analysis_id, timestamp=timestamp, **{
            k: v for k, v in result.items()
            if k in MultimodalAnalysisResult.model_fields and k not in ("id", "timestamp")
        }),
    )


# ──────────────────────────────────────────────
# History Endpoints
# ──────────────────────────────────────────────

@router.get("/history", response_model=HistoryListResponse)
async def list_history(
    limit: int = Query(20, ge=1, le=100),
    media_type: Optional[str] = Query(None),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """List recent analyses, scoped to the authenticated user."""
    user_id = current_user["id"] if current_user else None
    rows = await history.get_recent(limit=limit, media_type=media_type, user_id=user_id)
    entries = [
        HistoryEntry(**{k: v for k, v in row.items() if k in HistoryEntry.model_fields})
        for row in rows
    ]
    total = await history.count(user_id=user_id)
    return HistoryListResponse(success=True, data=entries, total=total)


@router.get("/history/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
):
    """Get a specific analysis result, scoped to the authenticated user."""
    user_id = current_user["id"] if current_user else None
    result = await history.get(analysis_id, user_id=user_id)
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
    """Health check endpoint — reports degraded when no models are loaded."""
    reg = get_registry()
    status = "healthy" if reg.loaded else "degraded"
    return HealthResponse(
        status=status,
        models_loaded=len(reg.loaded),
    )
