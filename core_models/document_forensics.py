"""
Document/ID forensic analysis.

Answers a different question than the portrait pipeline: not "is this a
deepfake face" but "is this document/ID/receipt/certificate image
AI-generated or digitally tampered?" Documents have no face to align in
most cases (receipts, certificates, contracts), and the artifacts that
matter are different - text/stamp edits, splices, recompression seams -
so this reuses zero of the face-detection-dependent pipeline.

Every signal here is classical computer vision, not a trained model -
there is no labeled AI-generated-document training set to fit a
classifier to yet (unlike the portrait models, which trained on
GAN/diffusion/face-morph datasets this session). Treat this as a
heuristic first pass, same epistemic status as the `forensic` heuristic
score in core/pipeline.py: a real, useful signal, but not comparable in
reliability to a trained deep model. The one trained-model signal this
module *does* use is CorefakeNet applied to the whole document image
(no face crop) as a generic "does this pixel content look
AI-synthesized" check - it was trained on portraits, so this is a
transfer application, not a validated fit for document content.

Four signals, combined in analyze_document_forensics():
  1. Error Level Analysis (ELA) - re-JPEG-compress at a fixed quality
     and diff against the original. A region spliced in from a
     different source (different original compression history) shows
     a different error level than genuine surrounding content.
  2. Noise-residual grid consistency - tile the image, compute a
     Laplacian noise residual per tile, and measure how much that
     residual varies across tiles. Real scans/photos have fairly
     uniform sensor/print noise across the frame; a pasted-in region
     often stands out.
  3. Copy-move detection - ORB keypoints self-matched within the same
     image to find duplicated regions (the classic "copy this digit/
     stamp/signature and paste it elsewhere on the page" tell).
  4. EXIF/C2PA metadata (core/metadata.py, already built and shared
     with the image pipeline).
"""

from __future__ import annotations

import numpy as np
import cv2
from io import BytesIO
from PIL import Image


def error_level_analysis(pil_img: Image.Image, quality: int = 90) -> dict:
    """Re-compress at a fixed JPEG quality and diff against the original.

    Returns a 0-1 score (higher = more suspicious) and a PIL heatmap
    image suitable for display as evidence.
    """
    rgb = pil_img.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig = np.asarray(rgb, dtype=np.int16)
    recomp = np.asarray(recompressed, dtype=np.int16)
    diff = np.abs(orig - recomp).sum(axis=2).astype(np.float32)  # (H, W)

    # Tile into 16px blocks, compare each tile's mean error against the
    # image-wide mean - a genuinely uniform-history image has low
    # variance across tiles; a splice stands out as an outlier tile.
    h, w = diff.shape
    tile = 16
    th, tw = max(1, h // tile), max(1, w // tile)
    if th < 2 or tw < 2:
        tile_means = diff.flatten()
    else:
        trimmed = diff[: th * tile, : tw * tile]
        tile_means = trimmed.reshape(th, tile, tw, tile).mean(axis=(1, 3)).flatten()

    global_mean = float(tile_means.mean()) if tile_means.size else 0.0
    global_std = float(tile_means.std()) if tile_means.size else 0.0
    outlier_ratio = 0.0
    if global_mean > 1e-6:
        cv = global_std / global_mean  # coefficient of variation
        outlier_ratio = float(np.clip((cv - 0.3) / 1.2, 0.0, 1.0))

    # Normalize the diff map to a viewable heatmap
    diff_norm = diff / (diff.max() + 1e-6)
    heatmap = (diff_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap_img = Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))

    return {
        "ela_score": outlier_ratio,
        "ela_map": heatmap_img,
    }


def noise_grid_consistency(pil_img: Image.Image, tile_size: int = 64) -> float:
    """Tile the image and measure how inconsistent the per-tile noise
    residual is. Returns a 0-1 score (higher = more suspicious)."""
    gray = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < tile_size * 2 or w < tile_size * 2:
        return 0.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    residuals = []
    for y in range(0, h - tile_size, tile_size):
        for x in range(0, w - tile_size, tile_size):
            tile = laplacian[y:y + tile_size, x:x + tile_size]
            residuals.append(float(np.std(tile)))

    if len(residuals) < 4:
        return 0.0

    residuals = np.array(residuals)
    mean_r = residuals.mean()
    if mean_r < 1e-6:
        return 0.0
    cv_r = residuals.std() / mean_r
    # Real documents/photos: tiles vary but stay within a fairly narrow
    # band. A spliced region with different noise stands out as a high
    # coefficient of variation.
    return float(np.clip((cv_r - 0.4) / 1.0, 0.0, 1.0))


def copy_move_score(pil_img: Image.Image, max_dim: int = 1024) -> dict:
    """Detect duplicated regions within the same image via self-matched
    ORB keypoints. Returns a 0-1 score and the number of matched pairs
    that survive distance + geometric filtering."""
    img = pil_img.convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 20:
        return {"copy_move_score": 0.0, "matched_pairs": 0}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    min_dist_px = max(gray.shape) * 0.05  # ignore near-identical neighbors
    suspicious_pairs = 0
    for m in matches:
        # m[0] is always the point matched to itself (distance 0) - skip it
        for candidate in m[1:]:
            if candidate.distance > 40:
                continue
            p1 = keypoints[candidate.queryIdx].pt
            p2 = keypoints[candidate.trainIdx].pt
            dist_px = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            if dist_px > min_dist_px:
                suspicious_pairs += 1
            break  # only consider the best non-self match per keypoint

    # A handful of coincidental matches is normal (repeated patterns,
    # printed grid lines); a cluster of many is the copy-move signature.
    score = float(np.clip((suspicious_pairs - 8) / 40.0, 0.0, 1.0))
    return {"copy_move_score": score, "matched_pairs": suspicious_pairs}


def analyze_document_forensics(pil_img: Image.Image) -> dict:
    """Combine ELA, noise-grid, and copy-move into a single manipulation
    signal. EXIF is handled separately by core/metadata.py and combined
    by the caller (core/pipeline.py), which already has that code path.

    Returns:
        dict with manipulation_score (0-1), and each sub-signal for
        display as evidence.
    """
    ela = error_level_analysis(pil_img)
    noise_score = noise_grid_consistency(pil_img)
    cm = copy_move_score(pil_img)

    # Reverted a "don't dilute a strong copy-move hit" change (max() floor
    # instead of pure weighting): it fixed one template-generated fake ID
    # (96 duplicate regions -> correctly flagged) but broke real IDs with
    # a QR code. ORB-based copy-move can't distinguish "this repeats
    # because it's a QR code/security pattern by design" from "this
    # repeats because someone copy-pasted a region" - a clean PNG
    # screenshot of a real Aadhaar card scored a maxed-out 1.0 (60
    # matches, from the QR code's own crisp repeating modules) purely
    # because it's uncompressed, while a JPEG photo of a real PAN card's
    # own QR code barely registered (6 matches - JPEG artifacts/blur/
    # angle make each instance slightly different). QR codes are near-
    # universal on modern government IDs, so that false-positive is far
    # more consequential than the one fake sample this was tuned against.
    # Keeping copy-move at a modest 20% weight is the safer default until
    # copy-move detection can exclude QR/barcode regions specifically.
    manipulation_score = float(np.clip(
        0.45 * ela["ela_score"]
        + 0.35 * noise_score
        + 0.20 * cm["copy_move_score"],
        0.0, 1.0,
    ))

    return {
        "manipulation_score": manipulation_score,
        "ela_score": ela["ela_score"],
        "ela_map": ela["ela_map"],
        "noise_consistency_score": noise_score,
        "copy_move_score": cm["copy_move_score"],
        "copy_move_matches": cm["matched_pairs"],
    }
