"""
EXIF metadata forensics and C2PA content credentials.

Provides non-neural forensic signals:
- EXIF tag analysis (camera, GPS, software, timestamps)
- Suspicion scoring for AI-generated images
- C2PA provenance chain reading (when c2pa-python installed)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from core.types import ExifMetadata

logger = logging.getLogger(__name__)

AI_SOFTWARE_KEYWORDS = [
    "stable diffusion", "dall-e", "midjourney", "comfyui",
    "automatic1111", "invoke ai", "novelai", "adobe firefly",
    "leonardo", "playground", "ideogram", "flux",
]

CAMERA_TAGS = {271: "Make", 272: "Model", 37386: "FocalLength"}
TIMESTAMP_TAGS = {36867: "DateTimeOriginal", 36868: "DateTimeDigitized", 306: "DateTime"}


def extract_exif(image: Image.Image) -> ExifMetadata:
    """Extract and analyze EXIF metadata from a PIL image."""
    suspicion_score = 0.0
    findings: list[str] = []
    raw: dict[str, str] = {}

    exif_data = image.getexif()

    if not exif_data:
        suspicion_score += 0.3
        findings.append("No EXIF metadata found (common in AI-generated images)")
        return ExifMetadata(
            has_exif=False,
            suspicious=True,
            suspicion_score=min(suspicion_score, 1.0),
            findings=findings,
            width=image.width,
            height=image.height,
        )

    # Parse all tags
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, str(tag_id))
        try:
            raw[tag_name] = str(value)
        except Exception:
            raw[tag_name] = "<unreadable>"

    # Camera make/model
    camera_make = raw.get("Make")
    camera_model = raw.get("Model")
    if not any(tag_id in exif_data for tag_id in CAMERA_TAGS):
        suspicion_score += 0.2
        findings.append("No camera make/model in EXIF")

    # Timestamps
    timestamp = None
    for tag_id in TIMESTAMP_TAGS:
        if tag_id in exif_data:
            timestamp = str(exif_data[tag_id])
            break
    if timestamp is None:
        suspicion_score += 0.1
        findings.append("No timestamp data in EXIF")

    # Software
    software = raw.get("Software", "")
    if software:
        software_lower = software.lower()
        for keyword in AI_SOFTWARE_KEYWORDS:
            if keyword in software_lower:
                suspicion_score += 0.4
                findings.append(f"AI software detected in EXIF: {software}")
                break
    else:
        suspicion_score += 0.05
        findings.append("No software tag in EXIF")

    # GPS data
    gps_coordinates = None
    try:
        gps_ifd = exif_data.get_ifd(0x8825)
        if gps_ifd:
            lat = _parse_gps_coord(gps_ifd, 2, 1)
            lon = _parse_gps_coord(gps_ifd, 4, 3)
            if lat is not None and lon is not None:
                gps_coordinates = f"{lat:.6f}, {lon:.6f}"
        else:
            suspicion_score += 0.1
            findings.append("No GPS data in EXIF")
    except Exception:
        findings.append("GPS data present but unreadable")

    # Orientation
    orientation = exif_data.get(274)

    suspicious = suspicion_score >= 0.3

    return ExifMetadata(
        camera_make=camera_make,
        camera_model=camera_model,
        timestamp=timestamp,
        software=software or None,
        gps_coordinates=gps_coordinates,
        orientation=orientation,
        width=image.width,
        height=image.height,
        has_exif=True,
        suspicious=suspicious,
        suspicion_score=min(suspicion_score, 1.0),
        findings=findings,
        raw=raw,
    )


def _parse_gps_coord(
    gps_ifd: dict, coord_tag: int, ref_tag: int
) -> Optional[float]:
    """Parse GPS coordinate from EXIF IFD."""
    coord = gps_ifd.get(coord_tag)
    ref = gps_ifd.get(ref_tag)
    if coord is None or ref is None:
        return None

    try:
        degrees = float(coord[0])
        minutes = float(coord[1])
        seconds = float(coord[2])
        result = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            result = -result
        return result
    except (TypeError, IndexError, ValueError):
        return None


def check_c2pa(file_path: str) -> dict[str, Any]:
    """
    Check for C2PA content credentials (provenance chain).

    Requires: pip install c2pa-python
    Returns trust signal if C2PA manifest found.
    """
    try:
        import c2pa  # type: ignore
    except ImportError:
        return {"has_c2pa": False, "available": False, "trust_boost": 0.0}

    try:
        reader = c2pa.Reader.from_file(file_path)
        manifest_json = reader.json()
        return {
            "has_c2pa": True,
            "available": True,
            "manifest": manifest_json,
            "trust_boost": -0.2,  # negative = reduces risk score
        }
    except Exception:
        return {"has_c2pa": False, "available": True, "trust_boost": 0.0}


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity verification."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_full_metadata(
    image: Image.Image, file_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Extract complete metadata dict for pipeline results.

    Combines EXIF analysis, C2PA check, and file info into
    the format expected by core/pipeline.py.
    """
    exif = extract_exif(image)

    metadata: dict[str, Any] = {
        "format": getattr(image, "format", None),
        "dimensions": [image.width, image.height],
        "mode": image.mode,
        "has_exif": exif.has_exif,
        "exif_suspicious": exif.suspicious,
        "exif_suspicion_score": exif.suspicion_score,
        "exif_findings": exif.findings,
    }

    if exif.has_exif:
        metadata["exif"] = {
            "camera_make": exif.camera_make,
            "camera_model": exif.camera_model,
            "timestamp": exif.timestamp,
            "software": exif.software,
            "gps": exif.gps_coordinates,
        }
    else:
        metadata["exif"] = None

    # C2PA check (requires file path)
    if file_path is not None:
        c2pa_result = check_c2pa(file_path)
        metadata["has_c2pa"] = c2pa_result["has_c2pa"]
        metadata["c2pa_trust_boost"] = c2pa_result["trust_boost"]
    else:
        metadata["has_c2pa"] = False
        metadata["c2pa_trust_boost"] = 0.0

    # File hash
    if file_path is not None:
        try:
            metadata["file_hash"] = compute_file_hash(file_path)[:16]
            metadata["file_size_bytes"] = Path(file_path).stat().st_size
        except (OSError, IOError):
            metadata["file_hash"] = None
            metadata["file_size_bytes"] = None
    else:
        metadata["file_hash"] = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        metadata["file_size_bytes"] = None

    return metadata
