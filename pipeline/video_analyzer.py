"""
Video deepfake detection pipeline.

Inspired by VeridisQuo's architecture:
  1. Extract frames at configurable FPS
  2. Detect faces per frame
  3. Run ensemble models on each frame
  4. Aggregate per-frame scores (majority voting / average)

Uses the same models already loaded by the app (DINOv2, EfficientNet, Face Deepfake).
"""

import cv2
import os
import tempfile
import torch
from PIL import Image
from torchvision import transforms

from pipeline.face_gate import face_present


# -------- Frame Extraction --------

def extract_frames(video_path, fps=1):
    """
    Extract frames from a video at the given sampling rate.

    Args:
        video_path: Path to video file.
        fps: Frames per second to extract (default 1).

    Yields:
        (frame_index, PIL.Image, timestamp_sec)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(video_fps / fps))

    frame_idx = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            # Convert BGR -> RGB -> PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            timestamp = frame_idx / video_fps
            yield extracted, pil_img, timestamp
            extracted += 1

        frame_idx += 1

    cap.release()


def get_video_info(video_path):
    """Return basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    duration = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    info["duration_sec"] = round(duration, 2)
    cap.release()
    return info


# -------- Per-Frame Analysis --------

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

WEIGHTS = {"dino": 0.4, "efficientnet": 0.35, "face": 0.25}
HIGH_CONFIDENCE_OVERRIDE = 0.9


def _analyze_single_frame(pil_img, dino_model, eff_model, face_model, device):
    """
    Run the ensemble on a single PIL image frame.
    Returns a dict with per-model scores and frame risk.
    """
    tensor = _transform(pil_img.convert("RGB")).unsqueeze(0).to(device)

    dino_prob = 0.0
    eff_prob = 0.0
    face_prob = 0.0
    has_face = False
    active_models = 0

    with torch.no_grad():
        if dino_model is not None:
            dino_prob = dino_model(tensor).item()
            active_models += 1

        if eff_model is not None:
            eff_prob = eff_model(tensor).item()
            active_models += 1

        # Face detection needs a file on disk
        if face_model is not None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                pil_img.save(tmp_path)
            try:
                has_face = face_present(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if has_face:
                real_prob = face_model(tensor).item()
                face_prob = 1.0 - real_prob
                active_models += 1

    if active_models == 0:
        return None

    # Weighted ensemble (same logic as image pipeline)
    total_weight = 0.0
    weighted_sum = 0.0
    if dino_model is not None:
        weighted_sum += WEIGHTS["dino"] * dino_prob
        total_weight += WEIGHTS["dino"]
    if eff_model is not None:
        weighted_sum += WEIGHTS["efficientnet"] * eff_prob
        total_weight += WEIGHTS["efficientnet"]
    if has_face and face_model is not None:
        weighted_sum += WEIGHTS["face"] * face_prob
        total_weight += WEIGHTS["face"]

    frame_risk = weighted_sum / total_weight if total_weight > 0 else 0.0

    max_prob = max(dino_prob, eff_prob, face_prob)
    if max_prob > HIGH_CONFIDENCE_OVERRIDE:
        frame_risk = max(frame_risk, max_prob)

    return {
        "dino_prob": round(dino_prob, 4),
        "eff_prob": round(eff_prob, 4),
        "face_prob": round(face_prob, 4),
        "has_face": has_face,
        "frame_risk": round(frame_risk, 4),
        "prediction": "FAKE" if frame_risk > 0.5 else "REAL",
        "active_models": active_models,
    }


# -------- Score Aggregation (VeridisQuo-style) --------

def aggregate_majority(frame_results):
    """Majority voting across frames."""
    fake_count = sum(1 for r in frame_results if r["prediction"] == "FAKE")
    real_count = len(frame_results) - fake_count
    prediction = "FAKE" if fake_count >= real_count else "REAL"
    confidence = max(fake_count, real_count) / len(frame_results)
    return prediction, confidence


def aggregate_average(frame_results):
    """Average risk score across frames."""
    avg_risk = sum(r["frame_risk"] for r in frame_results) / len(frame_results)
    prediction = "FAKE" if avg_risk > 0.5 else "REAL"
    return prediction, avg_risk


def aggregate_max_confidence(frame_results):
    """Return the prediction from the frame with highest risk."""
    max_frame = max(frame_results, key=lambda r: r["frame_risk"])
    return max_frame["prediction"], max_frame["frame_risk"]


# -------- Main Video Analysis --------

def analyze_video(video_path, dino_model, eff_model, face_model, device,
                  fps=1, aggregation="majority", progress_callback=None):
    """
    Full video deepfake detection pipeline.

    Args:
        video_path: Path to video file.
        dino_model, eff_model, face_model: Loaded models (or None).
        device: torch device.
        fps: Frame sampling rate (default 1 fps).
        aggregation: 'majority', 'average', or 'max' (default 'majority').
        progress_callback: Optional callable(current, total, message) for progress.

    Returns:
        dict with overall verdict and per-frame results.
    """
    info = get_video_info(video_path)
    if info is None:
        return {"error": "Cannot open video file."}

    # Estimate total frames to extract
    est_total = max(1, int(info["duration_sec"] * fps))

    frame_results = []
    for frame_idx, pil_img, timestamp in extract_frames(video_path, fps=fps):
        if progress_callback:
            progress_callback(
                frame_idx + 1, est_total,
                f"Analyzing frame {frame_idx + 1}/{est_total} ({timestamp:.1f}s)"
            )

        result = _analyze_single_frame(pil_img, dino_model, eff_model, face_model, device)
        if result is not None:
            result["frame_index"] = frame_idx
            result["timestamp"] = round(timestamp, 2)
            frame_results.append(result)

    if not frame_results:
        return {
            "error": "No frames could be analyzed. Ensure models are trained.",
            "video_info": info,
        }

    # Aggregate
    agg_funcs = {
        "majority": aggregate_majority,
        "average": aggregate_average,
        "max": aggregate_max_confidence,
    }
    agg_func = agg_funcs.get(aggregation, aggregate_majority)
    prediction, confidence = agg_func(frame_results)

    avg_risk = sum(r["frame_risk"] for r in frame_results) / len(frame_results)

    # Count faces detected across frames
    faces_detected = sum(1 for r in frame_results if r["has_face"])
    fake_frames = sum(1 for r in frame_results if r["prediction"] == "FAKE")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "avg_risk": round(avg_risk, 4),
        "aggregation_method": aggregation,
        "total_frames_analyzed": len(frame_results),
        "fake_frames": fake_frames,
        "real_frames": len(frame_results) - fake_frames,
        "faces_detected_in_frames": faces_detected,
        "video_info": info,
        "frame_results": frame_results,
    }
