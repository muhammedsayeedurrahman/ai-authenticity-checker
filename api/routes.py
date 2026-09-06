"""
FastAPI REST API endpoints for ProofyX.

Responses follow the envelope pattern: {success, data, error}.
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile,
)
from fastapi.responses import Response
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.schemas import (
    AudioAnalysisResponse, AudioAnalysisResult,
    DocumentAnalysisResponse, DocumentAnalysisResult,
    HealthResponse,
    HistoryEntry, HistoryListResponse,
    ImageAnalysisResponse, ImageAnalysisResult,
    ModelStatus,
    MultimodalAnalysisResponse, MultimodalAnalysisResult,
    VideoAnalysisResponse, VideoAnalysisResult,
)
from core.pipeline import (
    analyze_audio, analyze_document, analyze_image, analyze_multimodal,
    analyze_video, get_registry,
)
from core.auth import get_principal
from core.principal import Principal
from db.history import AnalysisHistory

logger = logging.getLogger(__name__)

router = APIRouter()
history = AnalysisHistory()
limiter = Limiter(key_func=get_remote_address)

# Serialize GPU inference — prevents concurrent model.forward() calls from
# corrupting CUDA state or producing wrong results.
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
TIMEOUT_MULTIMODAL = int(os.environ.get("PROOFYX_TIMEOUT_MULTIMODAL", "600"))

# Magic bytes for image format validation
MAGIC_BYTES: dict[str, list[bytes]] = {
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png": [b"\x89PNG"],
    ".webp": [b"RIFF"],
    ".bmp": [b"BM"],
    ".tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
}

# Fields to strip from results before returning to clients — gradcam_image
# is handled separately (serialized to base64) rather than stripped, since
# the frontend heatmap viewer needs it.
_STRIP_FIELDS = {"original_image"}


def _pil_to_base64(img: Image.Image) -> str:
    """Bare base64 PNG payload (no data-URI prefix — the frontend adds it)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
    """Read and validate an uploaded file (size + extension + magic bytes)."""
    ext = ""
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext and ext not in allowed_ext:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    contents = await file.read()
    if len(contents) > max_size:
        max_mb = max_size // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large. Maximum: {max_mb}MB")

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
    """Run a sync function in a thread pool with a timeout and GPU semaphore."""
    # Wait up to the same budget as the analysis itself to *acquire* the
    # semaphore, not a fixed 5s. With concurrency capped at 1 (see
    # _MAX_CONCURRENT_INFERENCE) and video/multimodal routinely taking well
    # over 5s - up to TIMEOUT_VIDEO=600s in ensemble mode - a second request
    # arriving while the first is still legitimately running used to get
    # rejected with "Server busy" after only 5s, even though the first
    # request wasn't stuck. Queuing behind it for the same budget the
    # analysis itself gets is the correct behavior; a genuinely wedged
    # request still eventually times out via the acquire wait_for below.
    try:
        await asyncio.wait_for(_inference_semaphore.acquire(), timeout=timeout)
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
        logger.warning("Analysis timed out after %ds: %s", timeout, fn.__name__)
        raise HTTPException(status_code=504, detail=f"Analysis timed out after {timeout}s")
    finally:
        _inference_semaphore.release()


async def _maybe_reverse_search(
    requested: bool, contents: bytes, filename: str,
) -> Optional[dict[str, Any]]:
    """Run reverse-image-search corroboration if the client opted in for
    this request. Not part of _run_with_timeout's GPU semaphore - this is
    an outbound network call to a third-party provider, unrelated to
    local inference concurrency, so it gets its own short timeout."""
    if not requested:
        return None

    from core_models.reverse_search import reverse_image_search

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(reverse_image_search, contents, filename),
            timeout=20,
        )
    except asyncio.TimeoutError:
        return {"available": True, "provider": "bing_visual_search", "matches": [],
                "match_count": 0, "error": "timed out"}


async def _save_and_build_response(
    pipeline_result: dict[str, Any],
    file_name: str,
    user_id: Optional[str],
    result_model: type,
    org_id: Optional[str] = None,
) -> tuple[str, str, dict[str, Any]]:
    """Create a new record from pipeline output without mutating the original.

    Returns (analysis_id, timestamp, filtered_fields_dict).
    """
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build history record without mutating the pipeline result
    history_record = {
        **pipeline_result,
        "id": analysis_id,
        "timestamp": timestamp,
        "file_name": file_name,
    }
    await history.save(history_record, user_id=user_id, org_id=org_id)

    # Filter to only fields the response model accepts, stripping large blobs
    response_fields = {
        k: v
        for k, v in pipeline_result.items()
        if k in result_model.model_fields
        and k not in ("id", "timestamp")
        and k not in _STRIP_FIELDS
    }

    # GradCAM is a PIL Image in the raw pipeline result — the generic pass
    # above would pass it through unconverted (and fail Pydantic validation,
    # which expects a str). Overwrite with a base64-serialized version when
    # the response model declares the field (currently only ImageAnalysisResult).
    gradcam = pipeline_result.get("gradcam_image")
    if isinstance(gradcam, Image.Image) and "gradcam_image" in result_model.model_fields:
        response_fields["gradcam_image"] = _pil_to_base64(gradcam)

    return analysis_id, timestamp, response_fields


def _principal_ids(principal: Principal) -> tuple[Optional[str], Optional[str]]:
    """Extract (user_id, org_id) from a resolved Principal.

    user_id is only meaningful for JWT-authenticated callers, matching the
    legacy get_current_user contract (API-key/anonymous callers have no
    user identity, but may still carry org_id).
    """
    user_id = principal.user_id if principal.kind == "user" else None
    return user_id, principal.org_id


# ──────────────────────────────────────────────
# Analysis Endpoints
# ──────────────────────────────────────────────

@router.post("/analyze/image", response_model=ImageAnalysisResponse)
@limiter.limit("30/minute")
async def api_analyze_image(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Query("ensemble", pattern="^(ensemble|fast)$"),
    reverse_search: bool = Query(False, description="Opt-in: cross-reference this image against the public web via Bing Visual Search (sends the image to a third-party provider)"),
    principal: Principal = Depends(get_principal),
):
    """Analyze an uploaded image for deepfake indicators."""
    contents = await _read_validated(file, MAX_IMAGE_SIZE, ALLOWED_IMAGE_EXT)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = await _run_with_timeout(analyze_image, TIMEOUT_IMAGE, image, mode=mode)

    if result.get("error"):
        return ImageAnalysisResponse(success=False, error=result["error"])

    result["reverse_search"] = await _maybe_reverse_search(
        reverse_search, contents, file.filename or "image.jpg",
    )

    user_id, org_id = _principal_ids(principal)
    analysis_id, timestamp, fields = await _save_and_build_response(
        result, file.filename or "", user_id, ImageAnalysisResult, org_id=org_id,
    )

    return ImageAnalysisResponse(
        success=True,
        data=ImageAnalysisResult(id=analysis_id, timestamp=timestamp, **fields),
    )


@router.post("/analyze/document", response_model=DocumentAnalysisResponse)
@limiter.limit("30/minute")
async def api_analyze_document(
    request: Request,
    file: UploadFile = File(...),
    id_type: Optional[str] = Form(None, pattern="^(aadhaar|pan|voter_id|other)?$"),
    id_number: Optional[str] = Form(None),
    reverse_search: bool = Query(False, description="Opt-in: cross-reference this document against the public web via Bing Visual Search (sends the image to a third-party provider)"),
    principal: Principal = Depends(get_principal),
):
    """Analyze an uploaded document/ID/receipt/certificate image for AI
    generation or digital tampering. Image formats only in this version -
    PDFs would need a rasterization step (pdf2image/poppler) not yet
    wired in.

    id_type/id_number are optional: when both are provided, the number
    printed on the document (typed by the user, not OCR'd) is checked
    against the selected ID type's format/checksum - see
    core_models/id_validators.py. Neither field is required; omitting
    them just skips that one signal."""
    contents = await _read_validated(file, MAX_IMAGE_SIZE, ALLOWED_IMAGE_EXT)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid image file")

    suffix = ".jpg"
    if file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = await _run_with_timeout(
            analyze_document, TIMEOUT_IMAGE, image,
            file_path=tmp_path, id_type=id_type, id_number=id_number,
        )
    finally:
        _safe_tmp_remove(tmp_path)

    if result.get("error"):
        return DocumentAnalysisResponse(success=False, error=result["error"])

    result["reverse_search"] = await _maybe_reverse_search(
        reverse_search, contents, file.filename or "document.jpg",
    )

    user_id, org_id = _principal_ids(principal)
    analysis_id, timestamp, fields = await _save_and_build_response(
        result, file.filename or "", user_id, DocumentAnalysisResult, org_id=org_id,
    )

    return DocumentAnalysisResponse(
        success=True,
        data=DocumentAnalysisResult(id=analysis_id, timestamp=timestamp, **fields),
    )


_ID_PROOF_MAX_BYTES = 5 * 1024 * 1024  # matches the real portal's own 5MB cap
_ID_PROOF_MIME_TYPES = {"image/jpeg", "image/png"}


@router.post("/complaint/generate")
@limiter.limit("10/minute")
async def api_generate_complaint(
    request: Request,
    analysis: str = Form(...),
    file_name: str = Form(""),
    name: str = Form(...),
    phone: str = Form(""),
    email: str = Form(""),
    address: str = Form(""),
    incident_description: str = Form(""),
    id_proof: Optional[UploadFile] = File(None),
    principal: Principal = Depends(get_principal),
):
    """Generate a cyber crime complaint document from an analysis result
    already held by the client, for the user to review and file
    themselves. Never submits anything on the user's behalf — see
    core/cyber_complaint.py.

    Accepts multipart/form-data (rather than a JSON body) so an optional
    government ID proof image can be attached alongside the fields - the
    portal itself requires an ID proof upload (Aadhaar/PAN/Passport,
    JPG/PNG, under 5MB) in Complainant Details, so this mirrors that."""
    from core.cyber_complaint import generate_complaint_document

    if not name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    try:
        analysis_dict = json.loads(analysis)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="analysis must be valid JSON")

    complainant = {
        "name": name,
        "phone": phone,
        "email": email,
        "address": address,
        "incident_description": incident_description,
    }

    id_proof_payload = None
    if id_proof is not None:
        if id_proof.content_type not in _ID_PROOF_MIME_TYPES:
            raise HTTPException(status_code=400, detail="ID proof must be a JPG or PNG image")
        raw = await id_proof.read()
        if len(raw) > _ID_PROOF_MAX_BYTES:
            raise HTTPException(status_code=400, detail="ID proof must be under 5MB")
        id_proof_payload = {
            "data_uri": f"data:{id_proof.content_type};base64,{base64.b64encode(raw).decode('ascii')}",
            "filename": id_proof.filename or "id_proof",
        }

    try:
        content, mime_type, filename = generate_complaint_document(
            analysis_dict, complainant, file_name, id_proof_payload,
        )
    except Exception as e:  # Broad catch: document generation must never 500 opaquely
        logger.warning("Complaint document generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not generate complaint document")

    return Response(
        content=content,
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/analyze/video", response_model=VideoAnalysisResponse)
@limiter.limit("10/minute")
async def api_analyze_video(
    request: Request,
    file: UploadFile = File(...),
    fps: float = Query(1.0, ge=0.5, le=30),
    aggregation: str = Query("weighted_avg"),
    mode: str = Query("ensemble", pattern="^(ensemble|fast)$"),
    principal: Principal = Depends(get_principal),
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
            analyze_video, TIMEOUT_VIDEO, tmp_path,
            fps=fps, aggregation=aggregation, mode=mode,
        )
    finally:
        _safe_tmp_remove(tmp_path)

    if result.get("error"):
        return VideoAnalysisResponse(success=False, error=result["error"])

    user_id, org_id = _principal_ids(principal)
    analysis_id, timestamp, fields = await _save_and_build_response(
        result, file.filename or "", user_id, VideoAnalysisResult, org_id=org_id,
    )

    return VideoAnalysisResponse(
        success=True,
        data=VideoAnalysisResult(id=analysis_id, timestamp=timestamp, **fields),
    )


@router.post("/analyze/audio", response_model=AudioAnalysisResponse)
@limiter.limit("20/minute")
async def api_analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(get_principal),
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

    if result.get("error"):
        return AudioAnalysisResponse(success=False, error=result["error"])

    user_id, org_id = _principal_ids(principal)
    analysis_id, timestamp, fields = await _save_and_build_response(
        result, file.filename or "", user_id, AudioAnalysisResult, org_id=org_id,
    )

    return AudioAnalysisResponse(
        success=True,
        data=AudioAnalysisResult(id=analysis_id, timestamp=timestamp, **fields),
    )


@router.post("/analyze/multimodal", response_model=MultimodalAnalysisResponse)
@limiter.limit("10/minute")
async def api_analyze_multimodal(
    request: Request,
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    principal: Principal = Depends(get_principal),
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

    if result.get("error"):
        return MultimodalAnalysisResponse(success=False, error=result["error"])

    # Determine file_name from first available upload
    file_name = ""
    for upload in (image, video, audio):
        if upload is not None and upload.filename:
            file_name = upload.filename
            break

    user_id, org_id = _principal_ids(principal)
    analysis_id, timestamp, fields = await _save_and_build_response(
        result, file_name, user_id, MultimodalAnalysisResult, org_id=org_id,
    )

    return MultimodalAnalysisResponse(
        success=True,
        data=MultimodalAnalysisResult(id=analysis_id, timestamp=timestamp, **fields),
    )


# ──────────────────────────────────────────────
# History Endpoints
# ──────────────────────────────────────────────

@router.get("/history", response_model=HistoryListResponse)
async def list_history(
    limit: int = Query(20, ge=1, le=100),
    media_type: Optional[str] = Query(None),
    principal: Principal = Depends(get_principal),
):
    """List recent analyses, scoped to the caller's organization when the
    credential carries org context (org-scoped API key, or a JWT user with
    org membership), else scoped to the authenticated user.

    SECURITY: org-scoped API keys must never see another org's history —
    see _principal_ids and db/history.py::get_recent's org_id-priority
    scoping rule. Only a caller with neither org nor user identity (true
    legacy/dev-mode) receives unscoped results.
    """
    user_id, org_id = _principal_ids(principal)
    rows = await history.get_recent(limit=limit, media_type=media_type, user_id=user_id, org_id=org_id)
    entries = [
        HistoryEntry(**{k: v for k, v in row.items() if k in HistoryEntry.model_fields})
        for row in rows
    ]
    total = await history.count(user_id=user_id, org_id=org_id)
    return HistoryListResponse(success=True, data=entries, total=total)


@router.get("/history/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    principal: Principal = Depends(get_principal),
):
    """Get a specific analysis result, scoped to the caller's organization
    or user — see list_history's docstring for the scoping rule."""
    user_id, org_id = _principal_ids(principal)
    result = await history.get(analysis_id, user_id=user_id, org_id=org_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"success": True, "data": result}


# ──────────────────────────────────────────────
# System Endpoints
# ──────────────────────────────────────────────

@router.get("/models/status", response_model=ModelStatus)
async def models_status():
    """List loaded models and their status."""
    from core_models.reverse_search import is_configured as _reverse_search_configured

    reg = get_registry()
    return ModelStatus(
        **reg.get_status(),
        reverse_search_available=_reverse_search_configured(),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — reports degraded when no models are loaded."""
    reg = get_registry()
    status = "healthy" if reg.loaded else "degraded"
    return HealthResponse(status=status, models_loaded=len(reg.loaded))
