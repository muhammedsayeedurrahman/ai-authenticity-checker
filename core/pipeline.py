"""
Core detection pipeline for ProofyX.

Extracted from app.py — all functions return plain Python dicts
with no UI framework dependencies. Both the Gradio UI and FastAPI
REST API call these functions identically.

probability is ALWAYS P(fake): 0.0 = certainly real, 1.0 = certainly fake.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import logging
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from core.config import get_config
from core.types import (
    AnalysisResult, PredictionResult, Verdict, Confidence,
    RiskLevel, TemporalAnalysis, AudioResult,
)
from core.metadata import extract_full_metadata, extract_exif

logger = logging.getLogger(__name__)

# Standard ImageNet transform (shared across models)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────
# Model Registry (singleton)
# ──────────────────────────────────────────────

class ModelRegistry:
    """
    Loads all models once and provides inference methods.

    Replaces the scattered global variables in app.py:118-227.
    """

    def __init__(self):
        self.config = get_config()
        self.device = torch.device(self.config.device)
        self.models: dict[str, Any] = {}
        self.loaded: list[str] = []
        self.missing: list[str] = []

        # HuggingFace ViT
        self.vit_model = None
        self.vit_processor = None

        # Analyzers
        self.video_analyzer = None
        self.audio_analyzer = None
        self.freq_analyzer = None

        self._load_all()

    def _load_all(self) -> None:
        ROOT_DIR = self.config.models_dir.parent

        # Local PyTorch models
        self._try_load("dino", "DINOv2AuthModel",
                       "core_models.dinov2_auth_model", "dinov2_auth_model.pth")
        self._try_load("efficientnet", "EfficientNetAuthModel",
                       "core_models.efficientnet_auth_model", "efficientnet_auth_model.pth")
        self._try_load("face", "FaceDeepfakeModel",
                       "core_models.face_deepfake_model", "image_face_model.pth")
        self._try_load("texture", "EfficientNetTexture",
                       "core_models.efficientnet_texture", "efficient.pth")
        self._try_load("frequency", "FrequencyCNN",
                       "core_models.frequency_cnn", "frequency.pth")

        # FusionMLP (special: needs n_inputs arg)
        self._try_load_fusion()

        # CorefakeNet (special: checkpoint dict handling)
        self._try_load_corefakenet()

        # HuggingFace ViT
        self._try_load_vit()

        # Frequency analyzer (heuristic fallback)
        try:
            from pipeline.video_analyzer import FrequencyAnalyzer
            self.freq_analyzer = FrequencyAnalyzer()
        except ImportError:
            logger.warning("FrequencyAnalyzer not available")

        # Video & Audio analyzers
        self._init_video_analyzer()
        self._init_audio_analyzer()

        if self.loaded:
            logger.info("Loaded models: %s", ", ".join(self.loaded))
        if self.missing:
            logger.info("Missing models: %s", ", ".join(self.missing))

    def _try_load(self, name: str, class_name: str,
                  module_path: str, filename: str) -> None:
        path = self.config.models_dir / filename
        if not path.exists():
            self.missing.append(name)
            return
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            model = cls().to(self.device)
            model.load_state_dict(
                torch.load(str(path), map_location=self.device, weights_only=False)
            )
            model.eval()
            self.models[name] = model
            self.loaded.append(name)
        except Exception as e:
            logger.warning("Could not load %s: %s", name, e)
            self.missing.append(f"{name} (error)")

    def _try_load_fusion(self) -> None:
        path = self.config.models_dir / "fusion_mlp.pth"
        if not path.exists():
            self.missing.append("fusion")
            return
        try:
            from core_models.fusion_mlp import FusionMLP
            n_inputs = 4
            cfg = self.config.get_model("fusion")
            if cfg and cfg.n_inputs:
                n_inputs = cfg.n_inputs
            model = FusionMLP(n_inputs=n_inputs).to(self.device)
            model.load_state_dict(
                torch.load(str(path), map_location=self.device, weights_only=False)
            )
            model.eval()
            self.models["fusion"] = model
            self.loaded.append("fusion")
        except Exception as e:
            logger.warning("Could not load FusionMLP: %s", e)
            self.missing.append("fusion (error)")

    def _try_load_corefakenet(self) -> None:
        path = self.config.models_dir / "corefakenet.pth"
        if not path.exists():
            self.missing.append("corefakenet")
            return
        try:
            from core_models.corefakenet import CorefakeNet
            model = CorefakeNet().to(self.device)
            ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            self.models["corefakenet"] = model
            self.loaded.append("corefakenet")
            epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            logger.info("CorefakeNet loaded (epoch %s)", epoch)
        except Exception as e:
            logger.warning("Could not load CorefakeNet: %s", e)
            self.missing.append("corefakenet (error)")

    def _try_load_vit(self) -> None:
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            ROOT_DIR = self.config.models_dir.parent
            os.environ["HF_HOME"] = str(ROOT_DIR / ".hf_cache")
            cfg = self.config.get_model("vit")
            model_id = cfg.model_id if cfg else "prithivMLmods/Deep-Fake-Detector-v2-Model"
            self.vit_model = ViTForImageClassification.from_pretrained(model_id).to(self.device)
            self.vit_processor = ViTImageProcessor.from_pretrained(model_id)
            self.vit_model.eval()
            self.loaded.append("vit")
        except Exception as e:
            logger.warning("Could not load ViT: %s", e)
            self.missing.append("vit")

    def _init_video_analyzer(self) -> None:
        try:
            from pipeline.video_analyzer import VideoAnalyzer
            self.video_analyzer = VideoAnalyzer(
                dino_model=self.models.get("dino"),
                eff_model=self.models.get("efficientnet"),
                face_model=self.models.get("face"),
                device=self.device,
                vit_model=self.vit_model,
                vit_processor=self.vit_processor,
                texture_model=self.models.get("texture"),
                freq_cnn=self.models.get("frequency"),
                fusion_mlp=self.models.get("fusion"),
            )
        except Exception as e:
            logger.warning("Could not init VideoAnalyzer: %s", e)

    def _init_audio_analyzer(self) -> None:
        try:
            from pipeline.audio_analyzer import AudioAnalyzer
            self.audio_analyzer = AudioAnalyzer(device=self.device)
            if self.audio_analyzer.model_loaded:
                self.loaded.append("audio")
            else:
                self.missing.append("audio")
        except Exception as e:
            logger.warning("Could not init AudioAnalyzer: %s", e)

    def get_status(self) -> dict[str, Any]:
        return {
            "loaded": list(self.loaded),
            "missing": list(self.missing),
            "total": len(self.loaded),
            "corefakenet_ready": "corefakenet" in self.models,
        }


# ──────────────────────────────────────────────
# Module-level registry singleton
# ──────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


# ──────────────────────────────────────────────
# Score Helpers
# ──────────────────────────────────────────────

def calibrate_score(score: float, temperature: float = 1.2) -> float:
    """Apply temperature scaling for model score comparability."""
    score = max(min(score, 0.999), 0.001)
    logit = math.log(score / (1 - score))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def forensic_score(img_pil: Image.Image) -> float:
    """Detect manipulation via noise inconsistency and ELA."""
    import cv2
    from io import BytesIO

    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Noise inconsistency analysis
    patches = []
    patch_size = 64
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size].astype(np.float32)
            blur = cv2.GaussianBlur(patch, (5, 5), 0)
            noise = patch - blur
            patches.append(noise.std())

    if not patches:
        return 0.0

    noise_std = np.std(patches)
    noise_mean = np.mean(patches) + 1e-8
    noise_inconsistency = noise_std / noise_mean

    # Error Level Analysis
    buf = BytesIO()
    img_pil.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = np.array(Image.open(buf).convert("RGB")).astype(np.float32)
    original = img.astype(np.float32)
    ela_diff = np.abs(original - recompressed)
    ela_std = ela_diff.std()
    ela_score = min(ela_std / 20.0, 1.0)

    noise_score = min(max((noise_inconsistency - 0.5) / 0.6, 0.0), 1.0)
    return 0.6 * noise_score + 0.4 * ela_score


# ──────────────────────────────────────────────
# Image Analysis
# ──────────────────────────────────────────────

def analyze_image(image_pil: Image.Image, mode: str = "ensemble") -> dict[str, Any]:
    """
    Analyze a single image for deepfake indicators.

    Args:
        image_pil: PIL Image object
        mode: "ensemble" (7-model) or "fast" (CorefakeNet)

    Returns plain dict — see docs/ARCHITECTURE.md for schema.
    """
    start_time = time.perf_counter()
    reg = get_registry()

    if mode == "fast":
        return _analyze_image_fast(image_pil, reg, start_time)

    return _analyze_image_ensemble(image_pil, reg, start_time)


def _analyze_image_ensemble(
    image_pil: Image.Image, reg: ModelRegistry, start_time: float
) -> dict[str, Any]:
    """Full ensemble analysis (all models)."""
    from utils.gradcam import detect_and_align_face, generate_gradcam_image
    from core_models.frequency_cnn import fft_to_tensor
    from utils.explainability import explain_risk

    config = reg.config
    device = reg.device

    # Face alignment
    face_crop, face_bbox = detect_and_align_face(image_pil)
    has_face = face_crop is not None
    model_input = face_crop if has_face else image_pil
    tensor = TRANSFORM(model_input.convert("RGB")).unsqueeze(0).to(device)

    # Collect raw scores
    scores: dict[str, float] = {}
    active_models = 0

    with torch.no_grad():
        # ViT (HuggingFace)
        if reg.vit_model is not None and reg.vit_processor is not None:
            vit_inputs = reg.vit_processor(
                images=model_input.convert("RGB"), return_tensors="pt"
            ).to(device)
            vit_outputs = reg.vit_model(**vit_inputs)
            vit_probs = torch.softmax(vit_outputs.logits, dim=1)
            deepfake_idx = [
                k for k, v in reg.vit_model.config.id2label.items()
                if "fake" in v.lower() or "deep" in v.lower()
            ]
            scores["vit"] = (
                vit_probs[0][deepfake_idx[0]].item()
                if deepfake_idx else vit_probs[0][1].item()
            )
            active_models += 1

        # Texture (EfficientNet-B4)
        if "texture" in reg.models:
            scores["texture"] = reg.models["texture"](tensor).item()
            active_models += 1

        # Frequency CNN
        if "frequency" in reg.models:
            freq_input = face_crop if has_face else image_pil
            fft_tensor = fft_to_tensor(freq_input).unsqueeze(0).to(device)
            scores["frequency"] = reg.models["frequency"](fft_tensor).item()
            active_models += 1

        # DINOv2
        if "dino" in reg.models:
            scores["dino"] = reg.models["dino"](tensor).item()
            active_models += 1

        # EfficientNet Auth
        if "efficientnet" in reg.models:
            scores["efficientnet"] = reg.models["efficientnet"](tensor).item()
            active_models += 1

        # Face model (only when face detected)
        if has_face and "face" in reg.models:
            real_prob = reg.models["face"](tensor).item()
            scores["face"] = 1.0 - real_prob  # Convert P(real) → P(fake)
            active_models += 1

    # Forensic (heuristic)
    scores["forensic"] = forensic_score(image_pil)
    active_models += 1

    # Frequency fallback
    if "frequency" not in scores and reg.freq_analyzer:
        freq_input = face_crop if has_face else image_pil
        freq_result = reg.freq_analyzer.analyze(freq_input)
        scores["frequency"] = freq_result["frequency_score"]
        active_models += 1

    if active_models == 0:
        return _empty_result("image", start_time)

    # ── Fusion ──
    fusion_mode = "learned"
    fusion_mlp = reg.models.get("fusion")
    if fusion_mlp is not None:
        final_risk = fusion_mlp.predict(
            vit=scores.get("vit", 0.0),
            efficientnet=scores.get("texture", 0.0),
            forensic=scores.get("forensic", 0.0),
            frequency=scores.get("frequency", 0.0),
        )
    else:
        fusion_mode = "weighted_avg"
        cal = config.calibration
        cal_scores = {k: calibrate_score(v, cal.temperature) for k, v in scores.items()}

        use_boosted = (
            has_face and "face" in scores and scores["face"] > 0.6
        )
        weights = config.get_weights(face_boosted=use_boosted)

        total_weight = 0.0
        weighted_sum = 0.0
        for key, cal_val in cal_scores.items():
            if key in weights:
                weighted_sum += weights[key] * cal_val
                total_weight += weights[key]
        final_risk = weighted_sum / total_weight if total_weight > 0 else 0.0

        # High-confidence override
        trained_keys = ["vit", "texture", "face", "dino"]
        trained_cal = [cal_scores[k] for k in trained_keys if k in cal_scores]
        if trained_cal:
            max_prob = max(trained_cal)
            n_trained = len(trained_cal)
            override_thresh = cal.high_confidence_override if n_trained >= 3 else 0.50
            if max_prob > override_thresh:
                final_risk = max(final_risk, max_prob * 0.9 if n_trained < 3 else max_prob)

    # Verdict
    risk_pct = final_risk * 100
    verdict = Verdict.from_risk_score(final_risk)
    confidence = Confidence.from_risk_score(final_risk)
    risk_level = RiskLevel.from_risk_score(final_risk)

    # Model agreement
    fake_count = sum(1 for v in scores.values() if v > 0.5)
    model_agreement = f"{fake_count}/{active_models} models detect manipulation"

    # GradCAM
    gradcam_img = None
    try:
        gradcam_img = generate_gradcam_image(
            image_pil, reg.models.get("face"), device,
            vit_model=reg.vit_model, vit_processor=reg.vit_processor,
            eff_model=reg.models.get("efficientnet"),
            dino_model=reg.models.get("dino"),
        )
    except Exception as e:
        logger.warning("GradCAM failed: %s", e)

    # Explainability
    model_scores_for_explain = {
        "vit_prob": scores.get("vit", 0.0),
        "face_prob": scores.get("face", 0.0),
        "forensic_prob": scores.get("forensic", 0.0),
        "frequency_prob": scores.get("frequency", 0.0),
        "eff_prob": scores.get("texture", scores.get("efficientnet", 0.0)),
        "dino_prob": scores.get("dino", 0.0),
    }
    try:
        from utils.explainability import explain_risk
        explanation = explain_risk(final_risk, model_scores_for_explain)
    except Exception:
        explanation = ""

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "risk_score": final_risk,
        "risk_percent": risk_pct,
        "verdict": verdict.value,
        "confidence": confidence.value,
        "risk_level": risk_level.value,
        "model_agreement": model_agreement,
        "model_scores": scores,
        "fusion_mode": fusion_mode,
        "face_detected": has_face,
        "face_aligned": has_face,
        "gradcam_image": gradcam_img,
        "original_image": image_pil,
        "models_used": active_models,
        "processing_time_ms": elapsed_ms,
        "explanation": explanation,
        "media_type": "image",
    }


def _analyze_image_fast(
    image_pil: Image.Image, reg: ModelRegistry, start_time: float
) -> dict[str, Any]:
    """CorefakeNet single-model fast analysis."""
    from utils.gradcam import detect_and_align_face, generate_gradcam_image

    corefakenet = reg.models.get("corefakenet")
    if corefakenet is None:
        return _empty_result("image", start_time, error="CorefakeNet not loaded")

    face_crop, face_bbox = detect_and_align_face(image_pil)
    has_face = face_crop is not None
    model_input = face_crop if has_face else image_pil

    result = corefakenet.predict(model_input)

    final_risk = result["final_risk"]
    risk_pct = final_risk * 100
    verdict = Verdict.from_risk_score(final_risk)
    confidence_enum = Confidence.from_risk_score(final_risk)

    # Map CorefakeNet head scores
    scores = {}
    from core_models.corefakenet import CorefakeNet as CFN
    for name in CFN.HEAD_NAMES:
        scores[name] = result["model_scores"][f"{name}_score"]

    # Explainability
    model_scores_for_explain = {
        "vit_prob": scores.get("vit", 0.0),
        "face_prob": scores.get("artifact", 0.0),
        "forensic_prob": scores.get("frequency", 0.0),
        "frequency_prob": scores.get("frequency", 0.0),
        "eff_prob": scores.get("texture", 0.0),
        "dino_prob": scores.get("dino", 0.0),
    }
    try:
        from utils.explainability import explain_risk
        explanation = explain_risk(final_risk, model_scores_for_explain)
    except Exception:
        explanation = ""

    # GradCAM
    gradcam_img = None
    try:
        gradcam_img = generate_gradcam_image(
            image_pil, reg.models.get("face"), reg.device,
            vit_model=reg.vit_model, vit_processor=reg.vit_processor,
            eff_model=reg.models.get("efficientnet"),
            dino_model=reg.models.get("dino"),
        )
    except Exception:
        pass

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "risk_score": final_risk,
        "risk_percent": risk_pct,
        "verdict": verdict.value,
        "confidence": confidence_enum.value,
        "risk_level": RiskLevel.from_risk_score(final_risk).value,
        "model_agreement": f"CorefakeNet (5 heads, attention-fused)",
        "model_scores": scores,
        "fusion_mode": "corefakenet_attention",
        "face_detected": has_face,
        "face_aligned": has_face,
        "gradcam_image": gradcam_img,
        "original_image": image_pil,
        "models_used": 1,
        "processing_time_ms": elapsed_ms,
        "explanation": explanation,
        "media_type": "image",
        "corefakenet_details": {
            "attention_weights": result.get("attention_weights", {}),
            "temperature": result.get("temperature", 0.0),
            "confidence_raw": result.get("confidence", 0.0),
        },
    }


# ──────────────────────────────────────────────
# Video Analysis
# ──────────────────────────────────────────────

def analyze_video(
    video_path: str,
    fps: float = 4.0,
    aggregation: str = "weighted_avg",
    progress_callback: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Analyze video for deepfake indicators.
    Returns plain dict — see docs/ARCHITECTURE.md for schema.
    """
    start_time = time.perf_counter()
    reg = get_registry()

    if reg.video_analyzer is None:
        return _empty_result("video", start_time, error="VideoAnalyzer not available")

    def _progress(current, total, message):
        if progress_callback:
            progress_callback(current, total, message)

    result = reg.video_analyzer.analyze(
        video_path=video_path,
        fps=fps,
        aggregation=aggregation,
        progress_callback=_progress,
    )

    if "error" in result:
        return _empty_result("video", start_time, error=result["error"])

    risk_score = result["avg_risk"]
    risk_pct = risk_score * 100
    verdict = Verdict.from_risk_score(risk_score)
    confidence_enum = Confidence.from_risk_score(risk_score)

    temporal = result.get("temporal_summary", {})
    temporal_analysis = TemporalAnalysis(
        score_variance=temporal.get("overall_variance", 0.0),
        max_frame_jump=temporal.get("max_frame_jump", 0.0),
        significant_jumps=temporal.get("total_significant_jumps", 0),
        risk_timeline=[fr["frame_risk"] for fr in result.get("frame_results", [])],
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "risk_score": risk_score,
        "risk_percent": risk_pct,
        "verdict": verdict.value,
        "confidence": confidence_enum.value,
        "risk_level": RiskLevel.from_risk_score(risk_score).value,
        "prediction": result.get("prediction", "UNKNOWN"),
        "total_frames_analyzed": result.get("total_frames_analyzed", 0),
        "fake_frames": result.get("fake_frames", 0),
        "real_frames": result.get("real_frames", 0),
        "faces_detected_in_frames": result.get("faces_detected_in_frames", 0),
        "frame_results": result.get("frame_results", []),
        "temporal_analysis": {
            "score_variance": temporal_analysis.score_variance,
            "max_frame_jump": temporal_analysis.max_frame_jump,
            "significant_jumps": temporal_analysis.significant_jumps,
            "risk_timeline": temporal_analysis.risk_timeline,
        },
        "video_info": result.get("video_info", {}),
        "aggregation_method": result.get("aggregation_method", aggregation),
        "processing_time_ms": elapsed_ms,
        "media_type": "video",
    }


# ──────────────────────────────────────────────
# Audio Analysis
# ──────────────────────────────────────────────

def analyze_audio(
    audio_path: str,
    progress_callback: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Analyze audio for deepfake indicators.
    Returns plain dict — see docs/ARCHITECTURE.md for schema.
    """
    start_time = time.perf_counter()
    reg = get_registry()

    if reg.audio_analyzer is None:
        return _empty_result("audio", start_time, error="AudioAnalyzer not available")

    def _progress(current, total, message):
        if progress_callback:
            progress_callback(current, total, message)

    result = reg.audio_analyzer.analyze(
        audio_path=audio_path,
        progress_callback=_progress,
    )

    if "error" in result:
        return _empty_result("audio", start_time, error=result["error"])

    fake_prob = result.get("fake_probability", 0.0)
    auth_score = result.get("authenticity_score", 100.0)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "risk_score": fake_prob,
        "authenticity_score": auth_score,
        "verdict": Verdict.from_risk_score(fake_prob).value,
        "confidence": result.get("confidence", "MEDIUM"),
        "manipulation_type": result.get("manipulation_type", ""),
        "evidence": result.get("evidence", []),
        "segment_results": result.get("segment_results", []),
        "suspicious_timestamps": result.get("timestamps", []),
        "duration_sec": result.get("duration_sec", 0.0),
        "segments_analyzed": result.get("segments_analyzed", 0),
        "processing_time_ms": elapsed_ms,
        "media_type": "audio",
        "explanation": result.get("explanation", ""),
    }


# ──────────────────────────────────────────────
# Multimodal Fusion
# ──────────────────────────────────────────────

def analyze_multimodal(
    image: Optional[Image.Image] = None,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Cross-modal fusion analysis.
    Returns plain dict — see docs/ARCHITECTURE.md for schema.
    """
    start_time = time.perf_counter()
    results: dict[str, dict] = {}
    modality_scores: dict[str, Optional[float]] = {
        "image": None, "video": None, "audio": None,
    }

    if image is not None:
        img_result = analyze_image(image, mode="ensemble")
        if "error" not in img_result:
            results["image"] = img_result
            modality_scores["image"] = img_result["risk_score"]

    if video_path is not None:
        vid_result = analyze_video(video_path)
        if "error" not in vid_result:
            results["video"] = vid_result
            modality_scores["video"] = vid_result["risk_score"]

    if audio_path is not None:
        aud_result = analyze_audio(audio_path)
        if "error" not in aud_result:
            results["audio"] = aud_result
            modality_scores["audio"] = aud_result["risk_score"]

    active = {k: v for k, v in results.items()}
    if not active:
        return _empty_result("multimodal", start_time, error="No media provided")

    # Weighted fusion
    active_scores = {k: v["risk_score"] for k, v in active.items()}
    fusion_weights = _compute_fusion_weights(set(active_scores.keys()))
    final_score = sum(
        fusion_weights[k] * active_scores[k] for k in active_scores
    )

    verdict = Verdict.from_risk_score(final_score)
    confidence_enum = Confidence.from_risk_score(final_score)

    try:
        from utils.explainability import explain_multimodal
        explanation = explain_multimodal(
            {k: round(v * 100, 1) if v is not None else None
             for k, v in modality_scores.items()},
            final_score,
        )
    except Exception:
        explanation = ""

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "risk_score": final_score,
        "risk_percent": final_score * 100,
        "verdict": verdict.value,
        "confidence": confidence_enum.value,
        "media_types": list(active.keys()),
        "modality_scores": modality_scores,
        "fusion_weights": fusion_weights,
        "explanation": explanation,
        "processing_time_ms": elapsed_ms,
        "media_type": "multimodal",
    }


def _compute_fusion_weights(modalities: set[str]) -> dict[str, float]:
    """Compute fusion weights based on available modalities."""
    weight_map = {
        frozenset({"image"}): {"image": 1.0},
        frozenset({"video"}): {"video": 1.0},
        frozenset({"audio"}): {"audio": 1.0},
        frozenset({"image", "video"}): {"image": 0.5, "video": 0.5},
        frozenset({"image", "audio"}): {"image": 0.6, "audio": 0.4},
        frozenset({"video", "audio"}): {"video": 0.6, "audio": 0.4},
        frozenset({"image", "video", "audio"}): {"image": 0.35, "video": 0.35, "audio": 0.3},
    }
    key = frozenset(modalities)
    if key in weight_map:
        return weight_map[key]
    # Equal weighting fallback
    n = len(modalities)
    return {m: 1.0 / n for m in modalities}


def _empty_result(media_type: str, start_time: float, error: str = "") -> dict[str, Any]:
    """Return empty/error result dict."""
    return {
        "risk_score": 0.0,
        "risk_percent": 0.0,
        "verdict": "",
        "confidence": "",
        "media_type": media_type,
        "processing_time_ms": (time.perf_counter() - start_time) * 1000,
        "error": error,
    }
