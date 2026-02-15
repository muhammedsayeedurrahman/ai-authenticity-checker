"""
Video deepfake detection pipeline (2026 redesign).

Modular architecture:
  1. VideoLoader      — frame extraction + metadata
  2. FaceExtractor    — face detection via OpenCV DNN
  3. FrequencyAnalyzer— FFT-based spectral heuristics
  4. ModelEnsemble    — weighted multi-model scoring
  5. TemporalAnalyzer — sliding-window temporal consistency
  6. VideoAnalyzer    — orchestrator
"""

import cv2
import os
import tempfile
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms
from collections import deque

from pipeline.face_gate import face_present


# -------- Shared transform --------
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
#  1. VideoLoader
# ================================================================

class VideoLoader:
    """Load video, extract metadata, and yield sampled frames."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.cap.release()

    @property
    def info(self):
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration, 2),
        }

    def extract_frames(self, sampling_fps=6):
        """
        Generator yielding (index, PIL.Image, timestamp).

        Args:
            sampling_fps: How many frames per second to sample.
                          Default 6 means ~every 5th frame of a 30fps video.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        skip = max(1, int(self.fps / sampling_fps))
        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                timestamp = frame_idx / self.fps
                yield extracted, pil_img, timestamp
                extracted += 1

            frame_idx += 1

        cap.release()


# ================================================================
#  2. FaceExtractor
# ================================================================

class FaceExtractor:
    """Face detection using OpenCV DNN (wraps face_gate.py)."""

    def __init__(self):
        self._last_bbox = None
        self._last_confidence = 0.0

    def detect(self, pil_img):
        """
        Detect face in a PIL image.

        Returns:
            (has_face, bbox, confidence)
            bbox is (x1, y1, x2, y2) in pixel coords or None.
        """
        # Save to temp file for OpenCV DNN
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_img.save(tmp_path)

        try:
            has_face, bbox, conf = self._detect_with_bbox(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        self._last_bbox = bbox
        self._last_confidence = conf
        return has_face, bbox, conf

    def _detect_with_bbox(self, image_path, confidence_threshold=0.5):
        """Run OpenCV DNN face detector and return bbox + confidence."""
        from pipeline.face_gate import _ensure_model_files, _PROTOTXT, _CAFFEMODEL
        _ensure_model_files()

        net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
        img = cv2.imread(image_path)
        if img is None:
            return False, None, 0.0

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        best_conf = 0.0
        best_box = None

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > confidence_threshold and confidence > best_conf:
                best_conf = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                best_box = (
                    max(0, x1), max(0, y1),
                    min(w, x2), min(h, y2),
                )

        if best_box is not None:
            return True, best_box, best_conf
        return False, None, 0.0

    def extract_face(self, pil_img, bbox=None):
        """
        Crop face region from PIL image.

        Args:
            pil_img: Source image.
            bbox: (x1, y1, x2, y2) or None to use last detection.

        Returns:
            Cropped PIL image or None.
        """
        if bbox is None:
            bbox = self._last_bbox
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        # Add 20% padding
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        img_w, img_h = pil_img.size
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)

        return pil_img.crop((x1, y1, x2, y2))


# ================================================================
#  3. FrequencyAnalyzer (FFT heuristic — no trained model)
# ================================================================

class FrequencyAnalyzer:
    """
    FFT-based spectral analysis for detecting AI-generated faces.

    AI-generated images typically show:
    - Less high-frequency detail (smoother textures)
    - Characteristic frequency dropoff patterns
    - GAN grid artifacts visible in angular spectrum
    """

    def analyze(self, pil_img):
        """
        Compute frequency-domain features for a face crop.

        Args:
            pil_img: PIL image (ideally a face crop).

        Returns:
            dict with individual scores and combined frequency_score (0-1).
        """
        img = np.array(pil_img.convert("L"), dtype=np.float32)

        # Resize to consistent size for comparable features
        img = cv2.resize(img, (256, 256))

        # 2D FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # --- High-frequency energy ratio ---
        high_freq_ratio = self._high_freq_energy_ratio(magnitude, cy, cx)

        # --- Radial power spectrum dropoff ---
        radial_dropoff = self._radial_dropoff_score(magnitude, cy, cx)

        # --- Azimuthal uniformity ---
        azimuthal_score = self._azimuthal_score(magnitude, cy, cx)

        # Combine: all scores are 0-1 where higher = more likely AI
        combined = (
            0.45 * high_freq_ratio +
            0.30 * radial_dropoff +
            0.25 * azimuthal_score
        )

        return {
            "high_freq_ratio": round(high_freq_ratio, 4),
            "radial_dropoff": round(radial_dropoff, 4),
            "azimuthal_score": round(azimuthal_score, 4),
            "frequency_score": round(combined, 4),
        }

    def _high_freq_energy_ratio(self, magnitude, cy, cx):
        """
        Ratio of high-frequency energy to total energy.
        AI images tend to have LESS high-frequency content.
        Lower ratio → more likely AI → higher score.
        """
        total_energy = magnitude.sum() + 1e-8
        h, w = magnitude.shape

        # Create radial distance map
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2)

        # High frequency = outer 40% of spectrum
        high_freq_mask = r > (0.6 * max_r)
        mid_freq_mask = (r > 0.2 * max_r) & (r <= 0.6 * max_r)

        high_energy = magnitude[high_freq_mask].sum()
        mid_energy = magnitude[mid_freq_mask].sum() + 1e-8

        ratio = high_energy / mid_energy

        # Real images typically have ratio > 0.3, AI < 0.25
        # Map: ratio 0.15 → score 0.9, ratio 0.35 → score 0.1
        score = 1.0 - np.clip((ratio - 0.10) / 0.30, 0.0, 1.0)
        return float(score)

    def _radial_dropoff_score(self, magnitude, cy, cx):
        """
        Measure how quickly the radial power spectrum drops off.
        AI images show steeper dropoff at high frequencies.
        """
        h, w = magnitude.shape
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        max_r = int(np.sqrt(cx ** 2 + cy ** 2))

        # Compute radial profile
        radial_profile = np.zeros(max_r + 1)
        radial_count = np.zeros(max_r + 1)
        r_flat = r.ravel()
        mag_flat = magnitude.ravel()

        for i in range(len(r_flat)):
            if r_flat[i] <= max_r:
                radial_profile[r_flat[i]] += mag_flat[i]
                radial_count[r_flat[i]] += 1

        radial_count[radial_count == 0] = 1
        radial_profile = radial_profile / radial_count

        if len(radial_profile) < 10:
            return 0.5

        # Compare high-freq band mean to mid-freq band mean
        n = len(radial_profile)
        mid_band = radial_profile[n // 4: n // 2]
        high_band = radial_profile[3 * n // 4:]

        if len(mid_band) == 0 or len(high_band) == 0:
            return 0.5

        mid_mean = mid_band.mean() + 1e-8
        high_mean = high_band.mean()
        dropoff_ratio = high_mean / mid_mean

        # Steeper dropoff (lower ratio) → more likely AI
        # ratio 0.2 → score 0.8, ratio 0.6 → score 0.2
        score = 1.0 - np.clip((dropoff_ratio - 0.15) / 0.50, 0.0, 1.0)
        return float(score)

    def _azimuthal_score(self, magnitude, cy, cx):
        """
        Measure angular uniformity of the spectrum.
        GAN artifacts create non-uniform angular patterns.
        """
        h, w = magnitude.shape
        y, x = np.mgrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        # Focus on mid-frequency band
        max_r = np.sqrt(cx ** 2 + cy ** 2)
        mask = (r > 0.2 * max_r) & (r < 0.6 * max_r)

        if mask.sum() < 10:
            return 0.5

        # Bin by angle
        n_bins = 36
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        bin_means = []

        for i in range(n_bins):
            angle_mask = (theta >= bin_edges[i]) & (theta < bin_edges[i + 1]) & mask
            if angle_mask.sum() > 0:
                bin_means.append(magnitude[angle_mask].mean())

        if len(bin_means) < 4:
            return 0.5

        bin_means = np.array(bin_means)
        # Coefficient of variation: higher = more non-uniform = more likely GAN
        cv = bin_means.std() / (bin_means.mean() + 1e-8)

        # cv > 0.15 suggests non-uniformity
        score = np.clip((cv - 0.05) / 0.25, 0.0, 1.0)
        return float(score)


# ================================================================
#  4. ModelEnsemble
# ================================================================

# Weights: ViT 30%, Frequency 20%, Forensic 20%, Face 15%, DINOv2 8%, EfficientNet 7%
WEIGHTS = {
    "vit": 0.30, "frequency": 0.20, "forensic": 0.20,
    "face": 0.15, "dino": 0.08, "efficientnet": 0.07,
}
# When face model detects fake (>0.6), boost face weight
WEIGHTS_FACE_BOOSTED = {
    "vit": 0.25, "frequency": 0.18, "forensic": 0.17,
    "face": 0.25, "dino": 0.08, "efficientnet": 0.07,
}
HIGH_CONFIDENCE_OVERRIDE = 0.65


class ModelEnsemble:
    """
    Holds all loaded models and computes weighted ensemble scores.
    """

    def __init__(self, dino_model, eff_model, face_model, device,
                 vit_model=None, vit_processor=None):
        self.dino_model = dino_model
        self.eff_model = eff_model
        self.face_model = face_model
        self.device = device
        self.vit_model = vit_model
        self.vit_processor = vit_processor
        self.frequency_analyzer = FrequencyAnalyzer()

    def predict(self, pil_img, has_face=False, face_crop=None):
        """
        Run all models on a single frame and return per-model scores.

        Args:
            pil_img: Full frame as PIL image.
            has_face: Whether a face was detected.
            face_crop: Cropped face PIL image (for frequency analysis).

        Returns:
            dict with per-model scores and final frame_risk.
        """
        tensor = _transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)

        dino_prob = 0.0
        eff_prob = 0.0
        face_prob = 0.0
        vit_prob = 0.0
        active_models = 0

        with torch.no_grad():
            if self.dino_model is not None:
                dino_prob = self.dino_model(tensor).item()
                active_models += 1

            if self.eff_model is not None:
                eff_prob = self.eff_model(tensor).item()
                active_models += 1

            if has_face and self.face_model is not None:
                real_prob = self.face_model(tensor).item()
                face_prob = 1.0 - real_prob
                active_models += 1

            if self.vit_model is not None and self.vit_processor is not None:
                vit_inputs = self.vit_processor(
                    images=pil_img.convert("RGB"), return_tensors="pt"
                ).to(self.device)
                vit_outputs = self.vit_model(**vit_inputs)
                vit_probs = torch.softmax(vit_outputs.logits, dim=1)
                deepfake_idx = [
                    k for k, v in self.vit_model.config.id2label.items()
                    if "fake" in v.lower() or "deep" in v.lower()
                ]
                vit_prob = (
                    vit_probs[0][deepfake_idx[0]].item()
                    if deepfake_idx else vit_probs[0][1].item()
                )
                active_models += 1

        # Forensic analysis (noise + ELA)
        forensic_prob = self._forensic_score(pil_img)
        active_models += 1

        # Frequency analysis (use face crop if available, else full frame)
        freq_input = face_crop if face_crop is not None else pil_img
        freq_result = self.frequency_analyzer.analyze(freq_input)
        freq_prob = freq_result["frequency_score"]
        active_models += 1

        if active_models == 0:
            return None

        # Weighted ensemble — boost face weight when face model detects fake
        use_boosted = has_face and self.face_model is not None and face_prob > 0.6
        w = WEIGHTS_FACE_BOOSTED if use_boosted else WEIGHTS

        total_weight = 0.0
        weighted_sum = 0.0

        if self.dino_model is not None:
            weighted_sum += w["dino"] * dino_prob
            total_weight += w["dino"]
        if self.eff_model is not None:
            weighted_sum += w["efficientnet"] * eff_prob
            total_weight += w["efficientnet"]
        if has_face and self.face_model is not None:
            weighted_sum += w["face"] * face_prob
            total_weight += w["face"]
        if self.vit_model is not None:
            weighted_sum += w["vit"] * vit_prob
            total_weight += w["vit"]
        weighted_sum += w["forensic"] * forensic_prob
        total_weight += w["forensic"]
        weighted_sum += w["frequency"] * freq_prob
        total_weight += w["frequency"]

        frame_risk = weighted_sum / total_weight if total_weight > 0 else 0.0

        # High-confidence override
        max_prob = max(dino_prob, eff_prob, face_prob, vit_prob, freq_prob)
        if max_prob > HIGH_CONFIDENCE_OVERRIDE:
            frame_risk = max(frame_risk, max_prob)

        return {
            "dino_prob": round(dino_prob, 4),
            "eff_prob": round(eff_prob, 4),
            "face_prob": round(face_prob, 4),
            "vit_prob": round(vit_prob, 4),
            "forensic_prob": round(forensic_prob, 4),
            "frequency_prob": round(freq_prob, 4),
            "freq_high_ratio": freq_result["high_freq_ratio"],
            "freq_radial_dropoff": freq_result["radial_dropoff"],
            "freq_azimuthal": freq_result["azimuthal_score"],
            "has_face": has_face,
            "frame_risk": round(frame_risk, 4),
            "prediction": "FAKE" if frame_risk > 0.5 else "REAL",
            "active_models": active_models,
        }

    @staticmethod
    def _forensic_score(pil_img):
        """Detect manipulation via noise inconsistency and ELA."""
        img = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

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

        buf = BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
        buf.seek(0)
        recompressed = np.array(Image.open(buf).convert("RGB")).astype(np.float32)
        original = img.astype(np.float32)
        ela_diff = np.abs(original - recompressed)
        ela_std = ela_diff.std()
        ela_score = min(ela_std / 20.0, 1.0)

        noise_score = min(max((noise_inconsistency - 0.5) / 0.6, 0.0), 1.0)
        return 0.6 * noise_score + 0.4 * ela_score


# ================================================================
#  5. TemporalAnalyzer
# ================================================================

class TemporalAnalyzer:
    """
    Sliding-window temporal consistency analysis.

    Tracks per-frame scores and detects:
    - Score variance (deepfakes show more frame-to-frame variation)
    - Temporal jumps (sudden changes in key model scores)
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self._scores = deque(maxlen=window_size)
        self._vit_scores = deque(maxlen=window_size)
        self._face_scores = deque(maxlen=window_size)
        self._freq_scores = deque(maxlen=window_size)

    def reset(self):
        self._scores.clear()
        self._vit_scores.clear()
        self._face_scores.clear()
        self._freq_scores.clear()

    def update(self, frame_result):
        """
        Add a frame result and compute temporal features.

        Args:
            frame_result: dict from ModelEnsemble.predict()

        Returns:
            dict with temporal metrics and adjustment factor.
        """
        risk = frame_result["frame_risk"]
        self._scores.append(risk)
        self._vit_scores.append(frame_result.get("vit_prob", 0.0))
        self._face_scores.append(frame_result.get("face_prob", 0.0))
        self._freq_scores.append(frame_result.get("frequency_prob", 0.0))

        scores = list(self._scores)
        n = len(scores)

        # Moving average
        moving_avg = np.mean(scores)

        # Score variance (higher = more inconsistent)
        score_variance = np.var(scores) if n > 1 else 0.0

        # Temporal jumps: max absolute difference between adjacent frames
        max_jump = 0.0
        jump_count = 0
        if n > 1:
            diffs = [abs(scores[i] - scores[i - 1]) for i in range(1, n)]
            max_jump = max(diffs)
            # Count significant jumps (>0.15 change)
            jump_count = sum(1 for d in diffs if d > 0.15)

        # Per-model temporal consistency
        vit_variance = np.var(list(self._vit_scores)) if n > 1 else 0.0
        face_variance = np.var(list(self._face_scores)) if n > 1 else 0.0

        # Temporal adjustment: boost risk when inconsistency is detected
        # Real videos should have very consistent scores across frames
        adjustment = 0.0
        if n >= 3:
            # High variance suggests manipulation
            if score_variance > 0.02:
                adjustment += min(score_variance * 2.0, 0.15)
            # Large jumps suggest splicing or face swap transitions
            if max_jump > 0.2:
                adjustment += min((max_jump - 0.2) * 0.5, 0.10)
            # High model-specific variance
            if vit_variance > 0.03:
                adjustment += min(vit_variance * 1.5, 0.10)

        adjustment = min(adjustment, 0.25)  # Cap total adjustment

        return {
            "moving_avg": round(moving_avg, 4),
            "score_variance": round(score_variance, 6),
            "max_jump": round(max_jump, 4),
            "jump_count": jump_count,
            "vit_variance": round(vit_variance, 6),
            "face_variance": round(face_variance, 6),
            "temporal_adjustment": round(adjustment, 4),
            "window_size": n,
        }


# ================================================================
#  6. VideoAnalyzer (orchestrator)
# ================================================================

class VideoAnalyzer:
    """
    Full video deepfake detection pipeline orchestrator.

    Composes VideoLoader, FaceExtractor, ModelEnsemble, and TemporalAnalyzer.
    """

    def __init__(self, dino_model, eff_model, face_model, device,
                 vit_model=None, vit_processor=None):
        self.ensemble = ModelEnsemble(
            dino_model, eff_model, face_model, device,
            vit_model=vit_model, vit_processor=vit_processor,
        )
        self.face_extractor = FaceExtractor()
        self.temporal = TemporalAnalyzer(window_size=10)

    def analyze(self, video_path, fps=6, aggregation="weighted_avg",
                progress_callback=None):
        """
        Full video analysis pipeline.

        Args:
            video_path: Path to video file.
            fps: Frame sampling rate (default 6 fps — every ~5th frame at 30fps).
            aggregation: 'weighted_avg' (default), 'majority', 'average', or 'max'.
            progress_callback: Optional callable(current, total, message).

        Returns:
            dict with overall verdict, per-frame results, temporal analysis.
        """
        loader = VideoLoader(video_path)
        info = loader.info
        self.temporal.reset()

        est_total = max(1, int(info["duration_sec"] * fps))

        frame_results = []
        temporal_snapshots = []

        for frame_idx, pil_img, timestamp in loader.extract_frames(sampling_fps=fps):
            if progress_callback:
                progress_callback(
                    frame_idx + 1, est_total,
                    f"Analyzing frame {frame_idx + 1}/{est_total} ({timestamp:.1f}s)"
                )

            # Face detection
            has_face, bbox, face_conf = self.face_extractor.detect(pil_img)
            face_crop = self.face_extractor.extract_face(pil_img, bbox) if has_face else None

            # Model ensemble prediction
            result = self.ensemble.predict(pil_img, has_face=has_face, face_crop=face_crop)
            if result is None:
                continue

            # Temporal analysis
            temporal = self.temporal.update(result)

            # Apply temporal adjustment to frame risk
            adjusted_risk = min(1.0, result["frame_risk"] + temporal["temporal_adjustment"])
            result["raw_risk"] = result["frame_risk"]
            result["frame_risk"] = round(adjusted_risk, 4)
            result["prediction"] = "FAKE" if adjusted_risk > 0.5 else "REAL"
            result["temporal_adjustment"] = temporal["temporal_adjustment"]

            result["frame_index"] = frame_idx
            result["timestamp"] = round(timestamp, 2)
            result["face_confidence"] = round(face_conf, 4)

            frame_results.append(result)
            temporal_snapshots.append(temporal)

        if not frame_results:
            return {
                "error": "No frames could be analyzed. Ensure models are trained.",
                "video_info": info,
            }

        # Aggregate
        prediction, confidence, avg_risk = self._aggregate(
            frame_results, aggregation
        )

        # Final temporal summary (from last window)
        final_temporal = temporal_snapshots[-1] if temporal_snapshots else {}

        # Overall temporal stats across ALL frames
        all_risks = [r["frame_risk"] for r in frame_results]
        overall_variance = float(np.var(all_risks)) if len(all_risks) > 1 else 0.0
        all_jumps = []
        for i in range(1, len(all_risks)):
            all_jumps.append(abs(all_risks[i] - all_risks[i - 1]))

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
            "temporal_summary": {
                "overall_variance": round(overall_variance, 6),
                "max_frame_jump": round(max(all_jumps) if all_jumps else 0.0, 4),
                "total_significant_jumps": sum(1 for j in all_jumps if j > 0.15),
                "final_window": final_temporal,
            },
        }

    def _aggregate(self, frame_results, method):
        """
        Aggregate per-frame scores into final verdict.

        Returns:
            (prediction, confidence, avg_risk)
        """
        risks = [r["frame_risk"] for r in frame_results]
        avg_risk = sum(risks) / len(risks)

        if method == "majority":
            fake_count = sum(1 for r in frame_results if r["prediction"] == "FAKE")
            real_count = len(frame_results) - fake_count
            prediction = "FAKE" if fake_count >= real_count else "REAL"
            confidence = max(fake_count, real_count) / len(frame_results)
            return prediction, confidence, avg_risk

        elif method == "max":
            max_frame = max(frame_results, key=lambda r: r["frame_risk"])
            prediction = max_frame["prediction"]
            confidence = max_frame["frame_risk"]
            return prediction, confidence, avg_risk

        elif method == "average":
            prediction = "FAKE" if avg_risk > 0.5 else "REAL"
            # confidence = |2 * risk - 1|
            confidence = abs(2 * avg_risk - 1)
            return prediction, confidence, avg_risk

        else:  # weighted_avg (default)
            prediction = "FAKE" if avg_risk > 0.5 else "REAL"
            # confidence = |fake_probability - real_probability|
            # = |avg_risk - (1 - avg_risk)| = |2 * avg_risk - 1|
            confidence = abs(2 * avg_risk - 1)
            return prediction, confidence, avg_risk


# ================================================================
#  Backward-compatible free functions (used by app.py GradCAM)
# ================================================================

def extract_frames(video_path, fps=1):
    """Backward-compatible wrapper for VideoLoader.extract_frames."""
    loader = VideoLoader(video_path)
    yield from loader.extract_frames(sampling_fps=fps)


def get_video_info(video_path):
    """Backward-compatible wrapper for VideoLoader.info."""
    try:
        loader = VideoLoader(video_path)
        return loader.info
    except ValueError:
        return None
