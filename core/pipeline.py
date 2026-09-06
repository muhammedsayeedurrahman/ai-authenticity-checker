"""
Core detection pipeline for ProofyX.

All functions return plain Python dicts with no UI framework
dependencies, called directly by the FastAPI REST API layer.

probability is ALWAYS P(fake): 0.0 = certainly real, 1.0 = certainly fake.
"""

from __future__ import annotations

import math
import os
import threading
import time
import logging
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from core.config import get_config
from core.types import (
    Verdict, Confidence,
    RiskLevel, TemporalAnalysis,
)

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

        # Classical ML tie-breaker (RandomForest on hand-crafted features)
        self.forensic_ml = None

        # HuggingFace models (new)
        self.wav2vec2_audio = None
        self.clip_deepfake = None

        # Analyzers
        self.video_analyzer = None
        self.audio_analyzer = None
        self.freq_analyzer = None
        self.fast_video_processor = None

        self._load_all()

    def _load_all(self) -> None:
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
        self._try_load("video_lstm", "VideoTemporalLSTM",
                       "core_models.video_lstm", "video_lstm.pth")

        # FusionMLP (special: needs n_inputs arg)
        self._try_load_fusion()

        # CorefakeNet (special: checkpoint dict handling)
        self._try_load_corefakenet()

        # HuggingFace ViT
        self._try_load_vit()

        # Classical ML tie-breaker
        self._try_load_forensic_ml()

        # Wav2Vec2 audio deepfake detector
        self._try_load_wav2vec2_audio()

        # CLIP ViT-L/14 deepfake detector
        self._try_load_clip_deepfake()

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
                torch.load(str(path), map_location=self.device, weights_only=True)
            )
            model.eval()
            self.models[name] = model
            self.loaded.append(name)
        except (RuntimeError, FileNotFoundError, ImportError, KeyError) as e:
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
                torch.load(str(path), map_location=self.device, weights_only=True)
            )
            model.eval()
            self.models["fusion"] = model
            self.loaded.append("fusion")
        except (RuntimeError, FileNotFoundError, ImportError, KeyError) as e:
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
            ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            self.models["corefakenet"] = model
            self.loaded.append("corefakenet")
            epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            logger.info("CorefakeNet loaded (epoch %s)", epoch)

            # Reuse the same in-memory model for video's "fast" mode instead
            # of loading a second copy from disk.
            from core_models.corefakenet import FastVideoProcessor
            self.fast_video_processor = FastVideoProcessor(
                model=model, device=str(self.device),
            )
        except (RuntimeError, FileNotFoundError, ImportError, KeyError) as e:
            logger.warning("Could not load CorefakeNet: %s", e)
            self.missing.append("corefakenet (error)")

    def _try_load_forensic_ml(self) -> None:
        path = self.config.models_dir / "forensic_ml.joblib"
        if not path.exists():
            self.missing.append("forensic_ml")
            return
        try:
            from core_models.forensic_ml import ForensicMLClassifier
            clf = ForensicMLClassifier()
            clf.load(str(path))
            self.forensic_ml = clf
            self.loaded.append("forensic_ml")
        except Exception as e:
            logger.warning("Could not load forensic ML tie-breaker: %s", e)
            self.missing.append("forensic_ml (error)")

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
        except (RuntimeError, FileNotFoundError, ImportError, KeyError) as e:
            logger.warning("Could not load ViT: %s", e)
            self.missing.append("vit")

    def _try_load_wav2vec2_audio(self) -> None:
        """Load Wav2Vec2-XLSR-300M audio deepfake detector from HuggingFace."""
        try:
            from core_models.wav2vec2_audio import Wav2Vec2AudioDetector
            cfg = self.config.get_model("wav2vec2_audio")
            model_id = cfg.model_id if cfg else None
            self.wav2vec2_audio = Wav2Vec2AudioDetector(
                device=self.device, model_id=model_id,
            )
            self.loaded.append("wav2vec2_audio")
        except (RuntimeError, FileNotFoundError, ImportError, KeyError, OSError) as e:
            logger.warning("Could not load Wav2Vec2 audio: %s", e)
            self.missing.append("wav2vec2_audio")

    def _try_load_clip_deepfake(self) -> None:
        """Load CLIP ViT-L/14 deepfake detector (TorchScript) from HuggingFace."""
        try:
            from core_models.clip_deepfake import CLIPDeepfakeDetector
            cache_dir = str(self.config.models_dir.parent / ".hf_cache")
            self.clip_deepfake = CLIPDeepfakeDetector(
                device=self.device, cache_dir=cache_dir,
            )
            self.loaded.append("clip")
        except (RuntimeError, FileNotFoundError, ImportError, KeyError, OSError) as e:
            logger.warning("Could not load CLIP deepfake: %s", e)
            self.missing.append("clip")

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
                video_lstm=self.models.get("video_lstm"),
                clip_deepfake=self.clip_deepfake,
            )
        except (RuntimeError, FileNotFoundError, ImportError, KeyError) as e:
            logger.warning("Could not init VideoAnalyzer: %s", e)

    def _init_audio_analyzer(self) -> None:
        try:
            from pipeline.audio_analyzer import AudioAnalyzer
            self.audio_analyzer = AudioAnalyzer(
                device=self.device,
                wav2vec2_model=self.wav2vec2_audio,
            )
            if self.audio_analyzer.model_loaded:
                self.loaded.append("audio")
            else:
                self.missing.append("audio")
        except (RuntimeError, FileNotFoundError, ImportError, KeyError) as e:
            logger.warning("Could not init AudioAnalyzer: %s", e)

    def get_status(self) -> dict[str, Any]:
        return {
            "loaded": list(self.loaded),
            "missing": list(self.missing),
            "total": len(self.loaded),
            "corefakenet_ready": "corefakenet" in self.models,
        }


# ──────────────────────────────────────────────
# Module-level registry singleton (thread-safe)
# ──────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """Return the shared ModelRegistry, creating it on first call.

    Uses double-checked locking so that concurrent threads never
    construct two registries or read a half-initialized instance.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry


# ──────────────────────────────────────────────
# Score Helpers
# ──────────────────────────────────────────────

def calibrate_score(score: float, temperature: float = 1.0) -> float:
    """Apply temperature scaling (Platt calibration) for model score comparability.

    NOTE: Temperatures are currently set to 1.0 (identity) because proper
    calibration requires fitting on a held-out validation set per model.
    When FusionMLP is loaded, its internal ModelCalibrator provides learned
    per-model temperatures instead — this function is only used as a
    fallback in the weighted_avg path.

    To properly calibrate:
        1. Collect model outputs on a held-out validation set
        2. Fit temperature via NLL minimization per model
        3. Update configs/models.json per_model_temperatures with fitted values
    """
    score = max(min(score, 0.999), 0.001)
    logit = math.log(score / (1 - score))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def _heuristic_forensic_score(img_pil: Image.Image) -> float:
    """Heuristic forensic signal based on noise inconsistency and ELA.

    WARNING: This is NOT a trained ML model. It uses hand-crafted feature
    extraction (Gaussian blur residual variance + JPEG Error Level Analysis)
    with hardcoded thresholds. It provides a weak supplementary signal but
    should NOT be weighted equally with trained models in the ensemble.

    When FusionMLP is loaded, this signal is fed as one of 7 inputs — the
    MLP was trained with these outputs and learns to weight them appropriately.
    In the weighted_avg fallback, this receives a reduced weight (0.05 vs 0.15+
    for trained models).
    """
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
    return float(0.6 * noise_score + 0.4 * ela_score)


# Public alias for backward compatibility (training scripts, tests)
forensic_score = _heuristic_forensic_score


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

        # CLIP ViT-L/14 deepfake detector
        if reg.clip_deepfake is not None:
            clip_score = float(reg.clip_deepfake.predict(model_input))
            scores["clip"] = clip_score
            active_models += 1

    # Forensic heuristic (NOT a trained model — supplementary signal only)
    scores["forensic"] = _heuristic_forensic_score(image_pil)

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
            texture=scores.get("texture", 0.0),
            forensic=scores.get("forensic", 0.0),
            frequency=scores.get("frequency", 0.0),
            dino=scores.get("dino", 0.0),
            efficientnet_auth=scores.get("efficientnet", 0.0),
            face=scores.get("face", 0.0),
        )
        # CLIP is NOT a FusionMLP input (7 fixed inputs) but used as
        # a post-fusion adjustment when CLIP has high confidence
        if "clip" in scores:
            clip_conf = abs(scores["clip"] - 0.5) * 2.0
            if clip_conf > 0.4:
                final_risk = 0.8 * final_risk + 0.2 * scores["clip"]
    else:
        fusion_mode = "weighted_avg"
        cal = config.calibration
        cal_scores = {
            k: calibrate_score(v, cal.per_model_temperatures.get(k, cal.temperature))
            for k, v in scores.items()
        }

        use_boosted = (
            has_face and "face" in scores and scores["face"] > 0.6
        )
        weights = config.get_weights(face_boosted=use_boosted)

        total_weight = 0.0
        weighted_sum = 0.0
        for key, cal_val in cal_scores.items():
            if key in weights:
                confidence = abs(cal_val - 0.5) * 2.0  # 0=uncertain, 1=confident
                effective_weight = weights[key] * (0.5 + 0.5 * confidence)
                weighted_sum += effective_weight * cal_val
                total_weight += effective_weight
        final_risk = weighted_sum / total_weight if total_weight > 0 else 0.0

        # High-confidence override
        trained_keys = ["vit", "texture", "face", "dino", "clip"]
        trained_cal = [cal_scores[k] for k in trained_keys if k in cal_scores]
        if trained_cal:
            max_prob = max(trained_cal)
            n_trained = len(trained_cal)
            override_thresh = cal.high_confidence_override if n_trained >= 3 else 0.50
            if max_prob > override_thresh:
                final_risk = max(final_risk, max_prob * 0.9 if n_trained < 3 else max_prob)

    # Classical-ML tie-breaker: hand-crafted forensic features (LBP, color
    # moments, noise residual, edge density) computed on the whole image,
    # face crop, and blend-boundary context region. Only consulted when the
    # fused score is near the decision boundary - training/diagnose_insight.py
    # found InsightFace-style face-swap-on-real-photo fakes clustering
    # exactly there (0.38-0.68), a different failure mode than the deep
    # ensemble being blind to them outright.
    forensic_ml_score = None
    if reg.forensic_ml is not None and 0.35 <= final_risk <= 0.75:
        try:
            forensic_ml_score = reg.forensic_ml.predict_proba_fake(image_pil)
            final_risk = 0.7 * final_risk + 0.3 * forensic_ml_score
        except Exception as e:
            logger.warning("Forensic ML tie-breaker failed: %s", e)

    # Verdict
    risk_pct = final_risk * 100
    verdict = Verdict.from_risk_score(final_risk)
    confidence = Confidence.from_risk_score(final_risk)
    risk_level = RiskLevel.from_risk_score(final_risk)

    # Model agreement (trained models only — excludes forensic heuristic)
    trained_scores = {k: v for k, v in scores.items() if k != "forensic"}
    fake_count = sum(1 for v in trained_scores.values() if v > 0.5)
    model_agreement = f"{fake_count}/{active_models} models detect manipulation"

    # GradCAM
    gradcam_overlay = None
    try:
        gradcam_img = generate_gradcam_image(
            image_pil, reg.models.get("face"), device,
            vit_model=reg.vit_model, vit_processor=reg.vit_processor,
            eff_model=reg.models.get("efficientnet"),
            dino_model=reg.models.get("dino"),
        )
        if gradcam_img is not None:
            from core.reports import image_to_base64
            gradcam_overlay = image_to_base64(gradcam_img)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("GradCAM failed: %s", e)

    # Explainability
    model_scores_for_explain = {
        "vit_prob": scores.get("vit", 0.0),
        "face_prob": scores.get("face", 0.0),
        "forensic_prob": scores.get("forensic", 0.0),
        "frequency_prob": scores.get("frequency", 0.0),
        "eff_prob": scores.get("texture", scores.get("efficientnet", 0.0)),
        "dino_prob": scores.get("dino", 0.0),
        "clip_prob": scores.get("clip"),
    }
    try:
        from utils.explainability import explain_risk
        explanation = explain_risk(final_risk, model_scores_for_explain)
    except (RuntimeError, ValueError, TypeError):
        explanation = ""

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Ensure all scores are native Python floats (not numpy/torch float32)
    scores = {k: float(v) for k, v in scores.items()}

    return {
        "risk_score": float(final_risk),
        "risk_percent": risk_pct,
        "verdict": verdict.value,
        "confidence": confidence.value,
        "risk_level": risk_level.value,
        "model_agreement": model_agreement,
        "model_scores": scores,
        "fusion_mode": fusion_mode,
        "forensic_ml_score": forensic_ml_score,
        "face_detected": has_face,
        "face_aligned": has_face,
        "gradcam_overlay": gradcam_overlay,
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
    except (RuntimeError, ValueError, TypeError):
        explanation = ""

    # GradCAM
    gradcam_overlay = None
    try:
        gradcam_img = generate_gradcam_image(
            image_pil, reg.models.get("face"), reg.device,
            vit_model=reg.vit_model, vit_processor=reg.vit_processor,
            eff_model=reg.models.get("efficientnet"),
            dino_model=reg.models.get("dino"),
        )
        if gradcam_img is not None:
            from core.reports import image_to_base64
            gradcam_overlay = image_to_base64(gradcam_img)
    except (RuntimeError, ValueError, TypeError):
        pass

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "risk_score": final_risk,
        "risk_percent": risk_pct,
        "verdict": verdict.value,
        "confidence": confidence_enum.value,
        "risk_level": RiskLevel.from_risk_score(final_risk).value,
        "model_agreement": "CorefakeNet (5 heads, attention-fused)",
        "model_scores": scores,
        "fusion_mode": "corefakenet_attention",
        "face_detected": has_face,
        "face_aligned": has_face,
        "gradcam_overlay": gradcam_overlay,
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

def _analyze_video_fast(
    video_path: str,
    fps: float,
    start_time: float,
    progress_callback: Optional[Callable] = None,
) -> dict[str, Any]:
    """Fast video path: single CorefakeNet pass/frame via FastVideoProcessor.

    Verdict/confidence/risk_level are computed the same standardized way as
    the ensemble path (Verdict.from_risk_score etc.) so callers can't tell
    which mode ran from those fields alone - only processing_time_ms and
    the smaller per-frame model_scores dict differ.
    """
    reg = get_registry()
    if reg.fast_video_processor is None:
        return _empty_result(
            "video", start_time,
            error="Fast video mode not available (CorefakeNet not loaded)",
        )

    def _progress(current, total, message):
        if progress_callback:
            progress_callback(current, total, message)

    # fps here means "samples per second" same as ensemble mode; fast mode's
    # single-model-per-frame cost is low enough that it doesn't need the
    # same aggressive downsampling, but still respect the caller's rate.
    result = reg.fast_video_processor.analyze(
        video_path, sampling_fps=max(fps, 0.5), progress_callback=_progress,
    )

    if "error" in result:
        return _empty_result("video", start_time, error=result["error"])

    risk_score = result["final_risk"]
    risk_pct = risk_score * 100
    verdict = Verdict.from_risk_score(risk_score)
    confidence_enum = Confidence.from_risk_score(risk_score)

    frame_results = result.get("frame_results", [])
    risk_timeline = [fr["final_risk"] for fr in frame_results]
    if len(risk_timeline) > 1:
        variance = float(np.var(risk_timeline))
        jumps = [abs(risk_timeline[i] - risk_timeline[i - 1])
                 for i in range(1, len(risk_timeline))]
        max_jump = max(jumps)
        significant_jumps = sum(1 for j in jumps if j > 0.15)
    else:
        variance, max_jump, significant_jumps = 0.0, 0.0, 0

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
        "frame_results": frame_results,
        "model_scores": result.get("model_scores", {}),
        "temporal_analysis": {
            "score_variance": variance,
            "max_frame_jump": max_jump,
            "significant_jumps": significant_jumps,
            "risk_timeline": risk_timeline,
        },
        "video_info": result.get("video_info", {}),
        "aggregation_method": "corefakenet_confidence_weighted",
        "fusion_mode": "corefakenet_fast",
        "processing_time_ms": elapsed_ms,
        "media_type": "video",
    }


def analyze_video(
    video_path: str,
    fps: float = 4.0,
    aggregation: str = "weighted_avg",
    mode: str = "ensemble",
    progress_callback: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Analyze video for deepfake indicators.
    Returns plain dict — see docs/ARCHITECTURE.md for schema.

    mode:
        "ensemble" (default) — 7-model ensemble per sampled frame. Thorough
            but slow on CPU (~10s/frame with all 7 models, including two
            transformers).
        "fast" — single CorefakeNet pass per frame via FastVideoProcessor
            (core_models/corefakenet.py). ~7x fewer forward passes per
            frame; combined with a lower sampling rate this is what makes
            a multi-minute analysis land in single-digit seconds.
    """
    start_time = time.perf_counter()
    reg = get_registry()

    if mode == "fast":
        return _analyze_video_fast(video_path, fps, start_time, progress_callback)

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
        "fusion_mode": "video_ensemble_7model",
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
        # "fast" mode (single lightweight CorefakeNet pass) was found to
        # miss a real deepfake that "ensemble" mode (7-model vote) caught
        # correctly (28% vs 65% risk on the same clip, confirmed via
        # direct testing) - a genuine accuracy gap, not just a sampling
        # density issue (raising fast mode's fps from 1.0 to 2.0 fixed a
        # separate run-to-run *consistency* bug but didn't fix this).
        # image already uses "ensemble" here for the same reason; video
        # was the odd one out. Ensemble is slower, but the overall
        # multimodal request already budgets up to 600s.
        vid_result = analyze_video(video_path, fps=1.0, mode="ensemble")
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

    # The fused risk_score/verdict is a single weighted number, which can
    # read as "everything here is fake" even when only some modalities
    # actually triggered - e.g. a genuine photo alongside a deepfaked
    # video+audio pair fuses to an AI-GENERATED verdict overall, which is
    # correct for "this submission contains AI-generated content" but
    # misleading if skimmed as "this image is fake too". Surfacing which
    # modalities individually crossed the threshold (same 0.60 cutoff
    # Verdict.from_risk_score uses) lets the UI say that explicitly
    # instead of only showing it buried in the per-modality score bars.
    flagged_modalities = sorted(
        k for k, v in active_scores.items() if v >= 0.60
    )
    clean_modalities = sorted(
        k for k, v in active_scores.items() if v < 0.60
    )

    try:
        from utils.explainability import explain_multimodal
        explanation = explain_multimodal(
            {k: round(v * 100, 1) if v is not None else None
             for k, v in modality_scores.items()},
            final_score,
        )
    except (RuntimeError, ValueError, TypeError):
        explanation = ""

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "risk_score": final_score,
        "risk_percent": final_score * 100,
        "verdict": verdict.value,
        "confidence": confidence_enum.value,
        "media_types": list(active.keys()),
        "modality_scores": modality_scores,
        "flagged_modalities": flagged_modalities,
        "clean_modalities": clean_modalities,
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


# ──────────────────────────────────────────────
# Document / ID Analysis
# ──────────────────────────────────────────────

def analyze_document(
    image_pil: Image.Image,
    file_path: Optional[str] = None,
    id_type: Optional[str] = None,
    id_number: Optional[str] = None,
) -> dict[str, Any]:
    """
    Analyze a document/ID/receipt/certificate image for AI generation or
    digital tampering. A distinct question from the portrait pipeline -
    most document content has no face to align, and the artifacts that
    matter (splices, recompression seams, copy-move edits) are different
    from deepfake face artifacts.

    Combines:
      - CorefakeNet applied to the whole image (no face crop) as a
        generic "does this look AI-synthesized" signal - a transfer
        application of a portrait-trained model, not a validated fit
        for document content, but the only trained-model signal
        available here.
      - core_models/document_forensics.py: ELA, noise-grid consistency,
        copy-move detection (classical CV, no training data needed).
      - core/metadata.py: EXIF + C2PA (already built, previously unused
        by any live endpoint).
      - core_models/id_validators.py: optional format/checksum check on
        a user-typed Aadhaar/PAN/Voter ID number (id_type/id_number) -
        not OCR, the user provides the number printed on the document.
        Aadhaar has a real public checksum (Verhoeff); PAN/Voter ID only
        have a documented format, no public checksum - see that module's
        docstring for why the two are treated differently.

    Returns plain dict — see docs/ARCHITECTURE.md for schema.
    """
    start_time = time.perf_counter()
    reg = get_registry()

    from core_models.document_forensics import analyze_document_forensics
    from core_models.id_validators import validate_id_number
    from core.metadata import extract_full_metadata
    from core.reports import image_to_base64

    img = image_pil.convert("RGB")

    # Generic AI-generation signal (whole image, no face crop).
    #
    # Tried cropping to an auto-detected embedded photo here on the theory
    # that CorefakeNet (portrait-trained) would score more accurately on a
    # face-like input than a whole document. Reverted: tested against a
    # real government ID, the tight crop (whether auto-detected or a
    # manual crop of just the photo) scored AI-GENERATED, while the whole
    # document correctly scored AUTHENTIC. A printed/laminated ID photo's
    # scan/print/recompression artifacts are themselves out-of-domain for
    # a model trained on natural camera photos vs. AI generations - cropping
    # tightly concentrates exactly that confusing artifact instead of
    # avoiding it. Whole-image is the empirically better input here, even
    # though it's still an unvalidated transfer application overall (see
    # the raised 0.72 decision threshold below).
    ai_generated_score = 0.0
    corefakenet = reg.models.get("corefakenet")
    if corefakenet is not None:
        with torch.no_grad():
            cfn_result = corefakenet.predict(img)
        ai_generated_score = float(cfn_result["final_risk"])

    forensics = analyze_document_forensics(img)
    manipulation_score = forensics["manipulation_score"]

    metadata = extract_full_metadata(img, file_path=file_path)
    exif_suspicion = metadata.get("exif_suspicion_score", 0.0)

    # AI-authoring-tool tag in EXIF is direct, strong evidence of
    # generation - boost ai_generated_score rather than diluting it into
    # the generic exif_suspicion blend.
    if metadata.get("exif") and any(
        "AI software detected" in f for f in metadata.get("exif_findings", [])
    ):
        ai_generated_score = max(ai_generated_score, 0.75)

    # C2PA content credentials: a structured, signed provenance chain is
    # stronger evidence than a free-text EXIF tag in either direction -
    # see core/metadata.py::check_c2pa for the three cases this covers.
    c2pa_info = metadata.get("c2pa") or {}
    if c2pa_info.get("ai_generated_signal"):
        ai_generated_score = max(ai_generated_score, 0.90)
    elif c2pa_info.get("validation_state") == "Invalid":
        manipulation_score = float(np.clip(
            manipulation_score + c2pa_info.get("trust_boost", 0.0), 0.0, 1.0,
        ))
    elif c2pa_info.get("valid"):
        manipulation_score = float(np.clip(
            manipulation_score + c2pa_info.get("trust_boost", 0.0), 0.0, 1.0,
        ))

    # Deliberately NOT folding generic exif_suspicion (missing camera/
    # timestamp/software/GPS tags) into manipulation_score here: every
    # photo of a physical document that's been through WhatsApp/a
    # messaging app - the overwhelmingly common real-world case for this
    # feature - gets its EXIF stripped by recompression whether the
    # document is genuine or not, so "no EXIF" is uninformative for
    # documents specifically (unlike for the general image pipeline,
    # where it's a real signal). The one exif finding that IS strong,
    # direct evidence - an AI-authoring-tool tag - is already handled
    # above by boosting ai_generated_score, not through this path.

    id_validation = validate_id_number(id_type, id_number)
    if id_validation is not None and id_validation["valid"] is False and id_type == "aadhaar":
        # A failed Verhoeff checksum is a real, deterministic defect -
        # unlike a PAN/Voter ID format mismatch (far more likely to be a
        # manual-entry typo than evidence of forgery), so this is the
        # only case that moves the score rather than staying purely
        # informational in the response.
        manipulation_score = float(np.clip(manipulation_score + 0.15, 0.0, 1.0))

    # Either signal alone is meaningful; averaging would dilute a real
    # hit from one detector with a quiet reading from the other (the
    # exact "dilution" failure mode documented for the portrait
    # ensemble - avoided here on purpose since these two signals
    # measure genuinely different things).
    risk_score = max(ai_generated_score, manipulation_score)
    confidence_enum = Confidence.from_risk_score(risk_score)

    # 0.60 is the calibrated threshold for CorefakeNet on its trained
    # domain (portraits). Applied blind to whole documents it is an
    # unvalidated transfer signal that misfires on genuine IDs/certificates
    # - holograms, lamination glare, and dense printed text read as
    # synthesis artifacts to a face-trained model. Raising the bar to 0.72
    # (still below the 0.75/0.90 EXIF/C2PA boosts above, which are direct
    # evidence and should keep triggering on their own) cuts obvious false
    # positives on genuine documents without needing retraining data this
    # repo doesn't have. Still a mitigation, not a fix - a document-domain
    # classifier trained on real ID/certificate data is the real fix.
    ai_generated_likely = ai_generated_score >= 0.72
    manipulation_likely = manipulation_score >= 0.60

    # The generic binary Verdict enum (AUTHENTIC/AI-GENERATED) can't
    # represent "flagged, but for tampering rather than AI generation" -
    # using it directly here mislabeled manipulation-only findings as
    # "AI-GENERATED" even when the AI-generation check itself came back
    # clean. Documents get a third, more honest label instead.
    if ai_generated_likely and ai_generated_score >= manipulation_score:
        primary_finding = "AI-GENERATED DOCUMENT SUSPECTED"
        verdict_label = "AI-GENERATED"
    elif manipulation_likely:
        primary_finding = "MANIPULATION SUSPECTED"
        verdict_label = "MANIPULATED"
    else:
        primary_finding = "AUTHENTIC DOCUMENT"
        verdict_label = "AUTHENTIC"

    checks = {
        "ai_generation": "Detected" if ai_generated_likely else "Not detected",
        "tampering": "Detected" if forensics["ela_score"] >= 0.5 else "Not detected",
        "copy_move": "Detected" if forensics["copy_move_score"] >= 0.5 else "Not detected",
        "metadata": "Suspicious" if metadata.get("exif_suspicious") else "Analyzed",
    }
    if id_validation is not None:
        checks["id_number"] = "Valid format" if id_validation["valid"] else "Invalid"
    if c2pa_info.get("has_c2pa"):
        if c2pa_info.get("ai_generated_signal"):
            checks["c2pa"] = "AI-generation declared"
        elif c2pa_info.get("validation_state") == "Invalid":
            checks["c2pa"] = "Tampered"
        elif c2pa_info.get("valid"):
            checks["c2pa"] = "Verified"
        else:
            checks["c2pa"] = "Present, unverified"

    evidence: dict[str, Any] = {}
    try:
        evidence["ela_map"] = image_to_base64(forensics["ela_map"])
    except Exception:  # Broad catch: evidence image is supplementary, never fatal
        pass

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "risk_score": risk_score,
        "risk_percent": risk_score * 100,
        "verdict": verdict_label,
        "confidence": confidence_enum.value,
        "risk_level": RiskLevel.from_risk_score(risk_score).value,
        "primary_finding": primary_finding,
        "ai_generated_likely": ai_generated_likely,
        "ai_generated_score": ai_generated_score,
        "manipulation_likely": manipulation_likely,
        "manipulation_score": manipulation_score,
        "checks": checks,
        "ela_score": forensics["ela_score"],
        "noise_consistency_score": forensics["noise_consistency_score"],
        "copy_move_score": forensics["copy_move_score"],
        "copy_move_matches": forensics["copy_move_matches"],
        "id_validation": id_validation,
        "evidence": evidence,
        "exif": {
            "has_exif": metadata.get("has_exif", False),
            "suspicious": metadata.get("exif_suspicious", False),
            "suspicion_score": exif_suspicion,
            "findings": metadata.get("exif_findings", []),
            "camera_make": (metadata.get("exif") or {}).get("camera_make"),
            "camera_model": (metadata.get("exif") or {}).get("camera_model"),
            "software": (metadata.get("exif") or {}).get("software"),
        },
        "has_c2pa": c2pa_info.get("has_c2pa", False),
        "c2pa": {
            "valid": c2pa_info.get("valid", False),
            "validation_state": c2pa_info.get("validation_state"),
            "generator": c2pa_info.get("generator"),
            "ai_generated_signal": c2pa_info.get("ai_generated_signal", False),
            "actions": c2pa_info.get("actions", []),
        } if c2pa_info.get("has_c2pa") else None,
        "processing_time_ms": elapsed_ms,
        "media_type": "document",
    }


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
