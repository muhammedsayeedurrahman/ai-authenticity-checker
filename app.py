import sys
import os
import math

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gradio as gr
import torch
import tempfile
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import json
import base64
from io import BytesIO
from typing import Optional
import re

from utils.explainability import explain_risk, explain_audio_risk, explain_multimodal
from utils.gradcam import (
    generate_gradcam_image, get_gradcam_for_face_model,
    get_gradcam_for_vit, get_gradcam_for_efficientnet,
    create_heatmap_overlay, create_face_region_overlay,
    detect_and_align_face, merge_heatmaps, _preprocess,
)
from pipeline.face_gate import face_present
from pipeline.video_analyzer import (
    VideoAnalyzer, extract_frames, get_video_info, FrequencyAnalyzer,
)
from pipeline.audio_analyzer import AudioAnalyzer

from core_models.dinov2_auth_model import DINOv2AuthModel
from core_models.efficientnet_auth_model import EfficientNetAuthModel
from core_models.face_deepfake_model import FaceDeepfakeModel
from core_models.efficientnet_texture import EfficientNetTexture
from core_models.frequency_cnn import FrequencyCNN, fft_to_tensor
from core_models.fusion_mlp import FusionMLP
from core_models.corefakenet import CorefakeNet, FastVideoProcessor

from transformers import ViTForImageClassification, ViTImageProcessor

# -------- Config --------
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Step 6: Corrected fusion weights — ViT 50%, EfficientNet 30%, Forensic 20%
#With auxiliary models (face, dino, frequency) contributing smaller shares
WEIGHTS = {
    "vit": 0.40, "efficientnet": 0.20, "forensic": 0.15,
    "frequency": 0.10, "face": 0.10, "dino": 0.05,
}
WEIGHTS_FACE_BOOSTED = {
    "vit": 0.35, "efficientnet": 0.15, "forensic": 0.15,
    "frequency": 0.10, "face": 0.20, "dino": 0.05,
}
HIGH_CONFIDENCE_OVERRIDE = 0.60  # Lowered: trust strong signals from any model


# -------- Score Calibration (Step 7) --------
def calibrate_score(score, temperature=1.2):
    """Apply temperature scaling for model score comparability.
    Temperature 1.2 preserves more signal than 1.5 (less compression toward 0.5).
    """
    score = max(min(score, 0.999), 0.001)
    logit = math.log(score / (1 - score))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


# -------- Forensic Analysis --------
def forensic_score(img_pil):
    """Detect manipulation via noise inconsistency and frequency analysis."""
    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    patches = []
    patch_size = 64
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size].astype(np.float32)
            blur = cv2.GaussianBlur(patch, (5, 5), 0)
            noise = patch - blur
            patches.append(noise.std())

    if not patches:
        return 0.0

    noise_std = np.std(patches)
    noise_mean = np.mean(patches) + 1e-8
    noise_inconsistency = noise_std / noise_mean

    from io import BytesIO
    buf = BytesIO()
    img_pil.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = np.array(Image.open(buf).convert("RGB")).astype(np.float32)
    original = img.astype(np.float32)
    ela_diff = np.abs(original - recompressed)
    ela_std = ela_diff.std()
    ela_score = min(ela_std / 20.0, 1.0)

    noise_score = min(max((noise_inconsistency - 0.5) / 0.6, 0.0), 1.0)
    combined = 0.6 * noise_score + 0.4 * ela_score

    return combined


VIT_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- Load models --------
dino_model = None
eff_model = None
face_model = None
loaded_models = []
missing_models = []


def try_load_model(name, model_class, filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        missing_models.append(name)
        return None
    try:
        model = model_class().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.eval()
        loaded_models.append(name)
        return model
    except Exception as e:
        print(f"WARNING: Could not load {name} from {filename}: {e}")
        missing_models.append(f"{name} (retrain needed)")
        return None


dino_model = try_load_model("DINOv2", DINOv2AuthModel, "dinov2_auth_model.pth")
eff_model = try_load_model("EfficientNet", EfficientNetAuthModel, "efficientnet_auth_model.pth")
face_model = try_load_model("Face Deepfake", FaceDeepfakeModel, "image_face_model.pth")

# New pipeline models (trained via train_all.py)
texture_model = try_load_model("EfficientNet-B4 Texture", EfficientNetTexture, "efficient.pth")
freq_cnn = try_load_model("Frequency CNN", FrequencyCNN, "frequency.pth")

fusion_mlp = None
fusion_path = os.path.join(MODELS_DIR, "fusion_mlp.pth")
if os.path.exists(fusion_path):
    try:
        fusion_mlp = FusionMLP(n_inputs=4).to(device)
        fusion_mlp.load_state_dict(torch.load(fusion_path, map_location=device, weights_only=False))
        fusion_mlp.eval()
        loaded_models.append("Fusion MLP")
    except Exception as e:
        print(f"WARNING: Could not load Fusion MLP from fusion_mlp.pth: {e}")
        fusion_mlp = None
        missing_models.append("Fusion MLP (retrain needed)")
else:
    missing_models.append("Fusion MLP")

# Load pre-trained ViT deepfake detector from HuggingFace
vit_model = None
vit_processor = None
try:
    os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
    vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_ID).to(device)
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_ID)
    vit_model.eval()
    loaded_models.append("ViT Deepfake Detector")
except Exception as e:
    print(f"Warning: Could not load ViT model: {e}")
    missing_models.append("ViT Deepfake Detector")

if loaded_models:
    print(f"Loaded models: {', '.join(loaded_models)}")
if missing_models:
    print(f"Missing model weights (train first): {', '.join(missing_models)}")

# -------- CorefakeNet (Fast Mode) --------
corefakenet_model = None
corefakenet_path = os.path.join(MODELS_DIR, "corefakenet.pth")
if os.path.exists(corefakenet_path):
    try:
        corefakenet_model = CorefakeNet().to(device)
        ckpt = torch.load(corefakenet_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            corefakenet_model.load_state_dict(ckpt['model_state_dict'])
        else:
            corefakenet_model.load_state_dict(ckpt)
        corefakenet_model.eval()
        loaded_models.append("CorefakeNet")
        print(f"CorefakeNet loaded (epoch {ckpt.get('epoch', '?')}, "
              f"val_acc={ckpt.get('best_val_acc', '?')})")
    except Exception as e:
        print(f"WARNING: Could not load CorefakeNet: {e}")
        corefakenet_model = None
        missing_models.append("CorefakeNet")
else:
    missing_models.append("CorefakeNet")

# -------- Frequency Analyzer (fallback when FrequencyCNN not available) --------
freq_analyzer = FrequencyAnalyzer()

# -------- Video Analyzer Instance --------
video_analyzer_instance = VideoAnalyzer(
    dino_model=dino_model,
    eff_model=eff_model,
    face_model=face_model,
    device=device,
    vit_model=vit_model,
    vit_processor=vit_processor,
    texture_model=texture_model,
    freq_cnn=freq_cnn,
    fusion_mlp=fusion_mlp,
)

# -------- Audio Analyzer Instance --------
audio_analyzer_instance = AudioAnalyzer(device=device)
if audio_analyzer_instance.model_loaded:
    loaded_models.append("Audio Deepfake CNN")
else:
    missing_models.append("Audio Deepfake CNN")


# -------- Image Prediction (Learned Fusion Pipeline) --------
def analyze_image(image):
    if image is None:
        return "", "", "", None

    # Save to temp file for face detection
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path)

    try:
        # Step 1: Face-aligned input
        face_crop, face_bbox = detect_and_align_face(image)
        has_face = face_crop is not None

        # Use face crop for all vision models when face is detected
        model_input = face_crop if has_face else image
        tensor = transform(model_input.convert("RGB")).unsqueeze(0).to(device)

        # --- Collect raw scores from all models ---
        vit_prob = 0.0
        texture_prob = 0.0
        freq_cnn_prob = 0.0
        dino_prob = 0.0
        eff_prob = 0.0
        face_prob = 0.0
        active_models = 0

        with torch.no_grad():
            # Primary models (used by FusionMLP)
            if vit_model is not None and vit_processor is not None:
                vit_inputs = vit_processor(
                    images=model_input.convert("RGB"), return_tensors="pt"
                ).to(device)
                vit_outputs = vit_model(**vit_inputs)
                vit_probs = torch.softmax(vit_outputs.logits, dim=1)
                deepfake_idx = [
                    k for k, v in vit_model.config.id2label.items()
                    if "fake" in v.lower() or "deep" in v.lower()
                ]
                vit_prob = (
                    vit_probs[0][deepfake_idx[0]].item()
                    if deepfake_idx else vit_probs[0][1].item()
                )
                active_models += 1

            if texture_model is not None:
                texture_prob = texture_model(tensor).item()
                active_models += 1

            if freq_cnn is not None:
                freq_input = face_crop if has_face else image
                fft_tensor = fft_to_tensor(freq_input).unsqueeze(0).to(device)
                freq_cnn_prob = freq_cnn(fft_tensor).item()
                active_models += 1

            # Auxiliary models (shown in details, not in fusion)
            if dino_model is not None:
                dino_prob = dino_model(tensor).item()
                active_models += 1

            if eff_model is not None:
                eff_prob = eff_model(tensor).item()
                active_models += 1

            if has_face and face_model is not None:
                real_prob = face_model(tensor).item()
                face_prob = 1.0 - real_prob
                active_models += 1

        # Forensic analysis (heuristic, on full frame)
        forensic_prob = forensic_score(image)
        active_models += 1

        # Fallback: heuristic frequency if FrequencyCNN not loaded
        freq_heuristic_prob = 0.0
        if freq_cnn is None:
            freq_input = face_crop if has_face else image
            freq_result = freq_analyzer.analyze(freq_input)
            freq_heuristic_prob = freq_result["frequency_score"]
            active_models += 1

        if active_models == 0:
            status = "No trained models found. Please train models first."
            details = "Run: python training/train_all.py"
            return status, details, "", None

        # --- Scoring: Learned Fusion MLP or fallback ---
        fusion_mode = "learned"
        if fusion_mlp is not None:
            # Use FusionMLP with learned calibration and weights
            # Input order: [vit, efficientnet_texture, forensic, frequency_cnn]
            final_risk = fusion_mlp.predict(
                vit=vit_prob,
                efficientnet=texture_prob,
                forensic=forensic_prob,
                frequency=freq_cnn_prob if freq_cnn is not None else freq_heuristic_prob,
            )
        else:
            # Fallback: manual weighted average with temperature calibration
            fusion_mode = "weighted_avg"
            cal_scores = {}
            if vit_model is not None:
                cal_scores["vit"] = calibrate_score(vit_prob)
            if texture_model is not None:
                cal_scores["efficientnet"] = calibrate_score(texture_prob)
            elif eff_model is not None:
                cal_scores["efficientnet"] = calibrate_score(eff_prob)
            if has_face and face_model is not None:
                cal_scores["face"] = calibrate_score(face_prob)
            if dino_model is not None:
                cal_scores["dino"] = calibrate_score(dino_prob)
            cal_scores["forensic"] = calibrate_score(forensic_prob)
            freq_val = freq_cnn_prob if freq_cnn is not None else freq_heuristic_prob
            cal_scores["frequency"] = calibrate_score(freq_val)

            use_boosted = has_face and face_model is not None and face_prob > 0.6
            w = WEIGHTS_FACE_BOOSTED if use_boosted else WEIGHTS

            total_weight = 0.0
            weighted_sum = 0.0
            for key, cal_val in cal_scores.items():
                if key in w:
                    weighted_sum += w[key] * cal_val
                    total_weight += w[key]
            final_risk = weighted_sum / total_weight if total_weight > 0 else 0.0

            # High-confidence override
            trained_cal = [cal_scores.get(k, 0) for k in ["vit", "efficientnet", "face", "dino"]
                           if k in cal_scores]
            if trained_cal:
                max_prob = max(trained_cal)
                n_trained = len(trained_cal)
                override_thresh = HIGH_CONFIDENCE_OVERRIDE if n_trained >= 3 else 0.50
                if max_prob > override_thresh:
                    final_risk = max(final_risk, max_prob * 0.9 if n_trained < 3 else max_prob)

        # Build explainability output
        freq_display = freq_cnn_prob if freq_cnn is not None else freq_heuristic_prob
        model_scores = {
            "vit_prob": vit_prob,
            "face_prob": face_prob,
            "forensic_prob": forensic_prob,
            "frequency_prob": freq_display,
            "eff_prob": texture_prob if texture_model is not None else eff_prob,
            "dino_prob": dino_prob,
        }
        verdict = explain_risk(final_risk, model_scores)
        risk_pct = final_risk * 100
        risk_label = f"AI-Generated Risk: {risk_pct:.1f}%"

        # Details
        details_lines = []
        details_lines.append(f"Face Detected      : {'Yes' if has_face else 'No'}")
        details_lines.append(f"Face-Aligned Input : {'Yes' if has_face else 'No (full frame)'}")
        details_lines.append(f"Fusion Mode        : {fusion_mode}")
        details_lines.append("")

        # Primary fusion inputs
        details_lines.append("--- Primary Models (Fusion Input) ---")
        if vit_model is not None:
            details_lines.append(f"ViT Deepfake       : {vit_prob:.4f}")
        else:
            details_lines.append(f"ViT Deepfake       : N/A (not loaded)")

        if texture_model is not None:
            details_lines.append(f"EfficientNet-B4 Tex: {texture_prob:.4f}")
        else:
            details_lines.append(f"EfficientNet-B4 Tex: N/A (not trained)")

        details_lines.append(f"Forensic (heuristic): {forensic_prob:.4f}")

        if freq_cnn is not None:
            details_lines.append(f"Frequency CNN      : {freq_cnn_prob:.4f}")
        else:
            details_lines.append(f"Frequency (heurist): {freq_heuristic_prob:.4f}")

        # Auxiliary models
        details_lines.append("")
        details_lines.append("--- Auxiliary Models ---")
        if dino_model is not None:
            details_lines.append(f"DINOv2 Auth        : {dino_prob:.4f}")
        if eff_model is not None:
            details_lines.append(f"EfficientNet Auth  : {eff_prob:.4f}")
        if has_face and face_model is not None:
            details_lines.append(f"Face Deepfake      : {face_prob:.4f}")

        details_lines.append("")
        details_lines.append(f"Final Risk Score   : {final_risk:.4f}")
        details_lines.append(f"Active Models      : {active_models}")

        if fusion_mlp is not None:
            temps = fusion_mlp.calibrator.temperatures.data.cpu().numpy()
            details_lines.append(f"Learned Temps      : vit={temps[0]:.2f} eff={temps[1]:.2f} "
                                 f"foren={temps[2]:.2f} freq={temps[3]:.2f}")

        details = "\n".join(details_lines)

        # Artifact-focused GradCAM (face-aligned, multi-model)
        gradcam_img = generate_gradcam_image(
            image, face_model, device,
            vit_model=vit_model, vit_processor=vit_processor,
            eff_model=eff_model, dino_model=dino_model,
        )

        return risk_label, details, verdict, gradcam_img

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# -------- Fast Image Prediction (CorefakeNet) --------
def analyze_image_fast(image):
    """Single-model analysis using CorefakeNet. ~5x faster than ensemble."""
    if image is None:
        return "", "", "", None

    if corefakenet_model is None:
        return ("CorefakeNet not loaded. Train with: python training/train_corefakenet.py",
                "", "", None)

    face_crop, face_bbox = detect_and_align_face(image)
    has_face = face_crop is not None
    model_input = face_crop if has_face else image

    result = corefakenet_model.predict(model_input)

    final_risk = result['final_risk']
    risk_pct = final_risk * 100
    risk_label = f"AI-Generated Risk: {risk_pct:.1f}%"

    model_scores = {
        "vit_prob": result['model_scores']['vit_score'],
        "face_prob": result['model_scores']['artifact_score'],
        "forensic_prob": result['model_scores']['frequency_score'],
        "frequency_prob": result['model_scores']['frequency_score'],
        "eff_prob": result['model_scores']['texture_score'],
        "dino_prob": result['model_scores']['dino_score'],
    }
    verdict = explain_risk(final_risk, model_scores)

    details_lines = []
    details_lines.append(f"Face Detected      : {'Yes' if has_face else 'No'}")
    details_lines.append(f"Face-Aligned Input : {'Yes' if has_face else 'No (full frame)'}")
    details_lines.append(f"Analysis Mode      : CorefakeNet (Fast)")
    details_lines.append(f"Fusion Mode        : attention-weighted (5 heads)")
    details_lines.append("")
    details_lines.append("--- CorefakeNet Head Scores ---")
    for name in CorefakeNet.HEAD_NAMES:
        score = result['model_scores'][f'{name}_score']
        details_lines.append(f"{name:12s}       : {score:.4f}")
    details_lines.append("")
    details_lines.append("--- Attention Weights ---")
    for name in CorefakeNet.HEAD_NAMES:
        w = result['attention_weights'][name]
        details_lines.append(f"{name:12s}       : {w:.4f}")
    details_lines.append("")
    details_lines.append(f"Temperature        : {result['temperature']:.4f}")
    details_lines.append(f"Confidence         : {result['confidence']:.4f}")
    details_lines.append(f"Final Risk Score   : {final_risk:.4f}")

    details = "\n".join(details_lines)

    # GradCAM using CorefakeNet backbone
    gradcam_img = None
    try:
        gradcam_img = generate_gradcam_image(
            image, face_model, device,
            vit_model=vit_model, vit_processor=vit_processor,
            eff_model=eff_model, dino_model=dino_model,
        )
    except Exception:
        pass

    return risk_label, details, verdict, gradcam_img


# -------- Image Prediction Router --------
def analyze_image_routed(image, mode):
    """Route to Fast Mode or Full Ensemble based on user selection."""
    if mode == "Fast Mode (CorefakeNet)":
        return analyze_image_fast(image)
    return analyze_image(image)


# -------- Video Prediction with face-aligned GradCAM (Step 5) --------
def analyze_video_ui(video, fps, aggregation, progress=gr.Progress()):
    if video is None:
        return "", "", "", None

    if progress:
        progress(0, desc="Starting video analysis...")

    def progress_callback(current, total, message):
        if progress:
            progress(current / max(total, 1), desc=message)

    result = video_analyzer_instance.analyze(
        video_path=video,
        fps=fps,
        aggregation=aggregation,
        progress_callback=progress_callback,
    )

    if "error" in result:
        return result["error"], "", "", None

    risk_pct = result["avg_risk"] * 100
    prediction = result["prediction"]
    confidence = result["confidence"] * 100

    risk_label = f"Verdict: {prediction} | AI Risk: {risk_pct:.1f}% | Confidence: {confidence:.1f}%"

    info = result["video_info"]
    temporal = result.get("temporal_summary", {})
    lines = []
    lines.append(f"Video Duration     : {info['duration_sec']}s ({info['width']}x{info['height']} @ {info['fps']:.1f} fps)")
    lines.append(f"Frames Analyzed    : {result['total_frames_analyzed']}")
    lines.append(f"Aggregation Method : {result['aggregation_method']}")
    lines.append("")
    lines.append(f"Fake Frames        : {result['fake_frames']}/{result['total_frames_analyzed']}")
    lines.append(f"Real Frames        : {result['real_frames']}/{result['total_frames_analyzed']}")
    lines.append(f"Faces Detected In  : {result['faces_detected_in_frames']}/{result['total_frames_analyzed']} frames")
    lines.append("")
    lines.append(f"Average Risk Score : {result['avg_risk']:.4f}")
    lines.append(f"Active Models      : {', '.join(loaded_models) if loaded_models else 'None'}")
    lines.append("")
    lines.append("--- Temporal Analysis ---")
    lines.append(f"Score Variance     : {temporal.get('overall_variance', 0):.6f}")
    lines.append(f"Max Frame Jump     : {temporal.get('max_frame_jump', 0):.4f}")
    lines.append(f"Significant Jumps  : {temporal.get('total_significant_jumps', 0)}")

    details = "\n".join(lines)

    # Per-frame breakdown
    frame_lines = []
    header = (
        f"{'Frame':<6} {'Time':>5} {'Risk':>6} {'Pred':>5} {'Face':>4} "
        f"{'ViT':>6} {'Freq':>6} {'Forns':>6} {'FaceM':>6} {'DINO':>6} {'Eff':>6} {'TAdj':>5}"
    )
    frame_lines.append(header)
    frame_lines.append("-" * len(header))
    for fr in result["frame_results"]:
        frame_lines.append(
            f"{fr['frame_index']:<6} {fr['timestamp']:>4.1f}s "
            f"{fr['frame_risk']:>5.3f} {fr['prediction']:>5} "
            f"{'Y' if fr['has_face'] else 'N':>4} "
            f"{fr['vit_prob']:>5.3f} {fr.get('frequency_prob', 0):>5.3f} "
            f"{fr['forensic_prob']:>5.3f} {fr['face_prob']:>5.3f} "
            f"{fr['dino_prob']:>5.3f} {fr['eff_prob']:>5.3f} "
            f"{fr.get('temporal_adjustment', 0):>+.2f}"
        )
    frame_details = "\n".join(frame_lines)

    # Step 5: Generate face-aligned GradCAM output video
    gradcam_video_path = None
    if face_model is not None or vit_model is not None:
        if progress:
            progress(0.9, desc="Generating face-aligned GradCAM video...")
        gradcam_video_path = _generate_gradcam_video(video, fps)

    return risk_label, details, frame_details, gradcam_video_path


def _generate_gradcam_video(video_path, fps):
    """Create side-by-side video with face-aligned GradCAM overlays."""
    info = get_video_info(video_path)
    if info is None:
        return None

    import imageio

    try:
        import imageio_ffmpeg
    except ImportError:
        print("WARNING: imageio[ffmpeg] not available. GradCAM video generation may fail.")

    frame_w, frame_h = 400, 400
    gap = 6
    canvas_w = frame_w * 2 + gap
    label_h = 40
    canvas_h = frame_h + label_h

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()

    out_fps = max(1, int(fps))

    try:
        writer = imageio.get_writer(
            tmp_out.name,
            format="FFMPEG",
            mode="I",
            fps=out_fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p"]
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        text_color = (0, 0, 0)

        label_left = "Original Frame"
        label_right = "Artifact Heatmap"
        left_size = cv2.getTextSize(label_left, font, font_scale, thickness)[0]
        right_size = cv2.getTextSize(label_right, font, font_scale, thickness)[0]

        prev_heatmap = None
        frame_count = 0

        for frame_idx, pil_img, timestamp in extract_frames(video_path, fps=fps):
            # Step 5: Face-aligned CAM per frame
            face_crop, face_bbox = detect_and_align_face(pil_img)
            model_input = face_crop if face_crop is not None else pil_img

            heatmap = None

            if face_model is not None:
                try:
                    tensor = _preprocess(model_input.convert("RGB")).unsqueeze(0).to(device)
                    heatmap = get_gradcam_for_face_model(face_model, tensor)
                except Exception:
                    pass

            if heatmap is None and vit_model is not None and vit_processor is not None:
                heatmap = get_gradcam_for_vit(vit_model, vit_processor, model_input, device)

            if heatmap is None:
                heatmap = prev_heatmap if prev_heatmap is not None else np.ones((7, 7)) * 0.2

            # Temporal smoothing
            if prev_heatmap is not None:
                h_resized = cv2.resize(heatmap, (224, 224))
                p_resized = cv2.resize(prev_heatmap, (224, 224))
                heatmap = 0.7 * h_resized + 0.3 * p_resized
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()

            prev_heatmap = heatmap

            # Overlay only on face region
            if face_bbox is not None:
                overlay_pil = create_face_region_overlay(
                    pil_img, heatmap, face_bbox, alpha=0.5, size=(frame_w, frame_h)
                )
            else:
                overlay_pil = create_heatmap_overlay(
                    pil_img, heatmap, alpha=0.5, size=(frame_w, frame_h)
                )

            original = np.array(pil_img.convert("RGB").resize((frame_w, frame_h)))
            overlay = np.array(overlay_pil)

            canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

            cv2.putText(canvas, label_left,
                         (frame_w // 2 - left_size[0] // 2, 28),
                         font, font_scale, text_color, thickness)
            cv2.putText(canvas, label_right,
                         (frame_w + gap + frame_w // 2 - right_size[0] // 2, 28),
                         font, font_scale, text_color, thickness)

            canvas[label_h:, :frame_w] = original
            canvas[label_h:, frame_w + gap:] = overlay

            writer.append_data(canvas)
            frame_count += 1

        writer.close()
        print(f"GradCAM Video generated | FPS: {out_fps} | Frames: {frame_count} | Output: {tmp_out.name}")
        return tmp_out.name

    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(tmp_out.name):
            try:
                os.unlink(tmp_out.name)
            except OSError:
                pass
        return None


# -------- Audio Prediction --------
def analyze_audio_ui(audio, progress=gr.Progress()):
    if audio is None:
        return "", "", ""

    if progress:
        progress(0, desc="Starting audio analysis...")

    def progress_callback(current, total, message):
        if progress:
            progress(current / max(total, 1), desc=message)

    result = audio_analyzer_instance.analyze(
        audio_path=audio,
        progress_callback=progress_callback,
    )

    if "error" in result:
        return result["error"], "", ""

    score = result["authenticity_score"]
    label = result["label"]
    confidence = result["confidence"]
    risk_label = f"Verdict: {label} | Authenticity: {score}% | Confidence: {confidence}"

    lines = []
    lines.append(f"Duration           : {result['duration_sec']}s")
    lines.append(f"Segments Analyzed  : {result['segments_analyzed']}")
    lines.append(f"Fake Probability   : {result['fake_probability']:.4f}")
    lines.append(f"Authenticity Score : {score}%")
    lines.append(f"Label              : {label}")
    lines.append(f"Confidence         : {confidence}")
    lines.append(f"Manipulation Type  : {result['manipulation_type']}")
    lines.append("")
    lines.append(f"Evidence           : {', '.join(result['evidence'])}")
    if result['timestamps']:
        lines.append(f"Suspicious Times   : {result['timestamps']}s")
    lines.append("")
    lines.append("--- Per-Segment Breakdown ---")
    for seg in result.get("segment_results", []):
        lines.append(
            f"  [{seg['start_time']:.1f}s - {seg['end_time']:.1f}s] "
            f"Fake: {seg['fake_probability']:.3f} | "
            f"Real: {seg['real_probability']:.3f}"
        )

    details = "\n".join(lines)
    verdict = result["explanation"]

    return risk_label, details, verdict


# -------- Multimodal Fusion (Step 6 fix) --------
def analyze_multimodal(image, video, audio, progress=gr.Progress()):
    """
    Multimodal routing and fusion with correct weight aggregation.
    """
    results = {}
    modality_scores = {"image": None, "video": None, "audio": None}

    if image is not None:
        if progress:
            progress(0.1, desc="Analyzing image...")
        risk_label, details, verdict, gradcam = analyze_image(image)
        try:
            score_str = risk_label.split(":")[1].strip().replace("%", "")
            img_score = float(score_str) / 100.0
        except (IndexError, ValueError):
            img_score = 0.5
        modality_scores["image"] = round(img_score * 100, 1)
        results["image"] = {"score": img_score, "details": details, "verdict": verdict}

    if video is not None:
        if progress:
            progress(0.3, desc="Analyzing video...")
        vid_result = video_analyzer_instance.analyze(video_path=video, fps=4, aggregation="weighted_avg")
        if "error" not in vid_result:
            vid_score = vid_result["avg_risk"]
            modality_scores["video"] = round(vid_score * 100, 1)
            results["video"] = {"score": vid_score, "prediction": vid_result["prediction"]}

    if audio is not None:
        if progress:
            progress(0.6, desc="Analyzing audio...")
        audio_result = audio_analyzer_instance.analyze(audio_path=audio)
        if "error" not in audio_result:
            audio_score = audio_result["fake_probability"]
            modality_scores["audio"] = round(audio_result["authenticity_score"], 1)
            results["audio"] = {"score": audio_score, "details": audio_result}

    active = {k: v for k, v in results.items()}
    if not active:
        return "No media provided", "{}", ""

    # Correct fusion with proper weights
    if len(active) == 1:
        key = list(active.keys())[0]
        final_score = active[key]["score"]
    elif "video" in active and "audio" in active and "image" not in active:
        # Fixed: was using video twice. Now: 0.6*video + 0.4*audio
        final_score = 0.6 * active["video"]["score"] + 0.4 * active["audio"]["score"]
    elif "video" in active and "audio" in active and "image" in active:
        final_score = 0.35 * active["image"]["score"] + 0.35 * active["video"]["score"] + 0.3 * active["audio"]["score"]
    elif "image" in active and "audio" in active:
        final_score = 0.6 * active["image"]["score"] + 0.4 * active["audio"]["score"]
    elif "image" in active and "video" in active:
        final_score = 0.5 * active["image"]["score"] + 0.5 * active["video"]["score"]
    else:
        scores = [v["score"] for v in active.values()]
        final_score = sum(scores) / len(scores)

    risk_pct = final_score * 100
    if final_score > 0.7:
        verdict = "HIGH RISK - Likely AI-generated/manipulated"
    elif final_score > 0.4:
        verdict = "MEDIUM RISK - Some manipulation indicators"
    else:
        verdict = "LOW RISK - Appears authentic"

    # Enhanced multimodal explanation
    mm_explanation = explain_multimodal(modality_scores, final_score)

    output_json = json.dumps({
        "media_types": list(active.keys()),
        "authenticity_score": round(100 - risk_pct, 1),
        "risk_score": round(risk_pct, 1),
        "label": verdict.split(" - ")[0],
        "modality_scores": modality_scores,
        "confidence": "High" if abs(final_score - 0.5) > 0.2 else "Medium",
        "explanation": mm_explanation,
    }, indent=2)

    risk_label = f"Multimodal Risk: {risk_pct:.1f}% | Modalities: {', '.join(active.keys())}"

    return risk_label, output_json, verdict


# -------- Helper Functions for API --------
def parse_model_scores(details_str):
    """Extract {name: float} from raw details text (Ensemble or CorefakeNet)."""
    scores = {}
    if not details_str:
        return scores
    for line in details_str.split("\n"):
        line = line.strip()
        if ":" not in line or line.startswith("---") or line.startswith("Face") or \
           line.startswith("Fusion") or line.startswith("Analysis") or \
           line.startswith("Active") or line.startswith("Learned") or \
           line.startswith("Final") or line.startswith("Temperature") or \
           line.startswith("Confidence"):
            continue
        parts = line.split(":")
        if len(parts) == 2:
            name = parts[0].strip()
            val_str = parts[1].strip()
            try:
                val = float(val_str)
                if 0 <= val <= 1:
                    scores[name] = val
            except ValueError:
                continue
    return scores


# ========== FASTAPI INTEGRATION ==========
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

fastapi_app = FastAPI(title="ProofyX API", description="Deepfake Detection API")

# Enable CORS for React frontend
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== API ENDPOINTS ==========
@fastapi_app.get("/api/status")
async def get_status():
    """Get model loading status"""
    return {
        "loaded_models": loaded_models,
        "missing_models": missing_models,
        "corefakenet_available": corefakenet_model is not None,
        "fusion_mlp_available": fusion_mlp is not None,
        "vit_available": vit_model is not None,
        "device": str(device)
    }


@fastapi_app.post("/api/analyze/image")
async def analyze_image_api(
    file: UploadFile = File(...),
    mode: str = Form("Full Ensemble (7 models)")
):
    """Analyze image for deepfakes"""
    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Call existing analysis function
        risk_label, details, verdict, gradcam_img = analyze_image_routed(image, mode)
        
        # Parse risk percentage
        try:
            risk_pct = float(risk_label.split(":")[1].strip().replace("%", ""))
        except (IndexError, ValueError):
            risk_pct = 0.0
        
        # Convert gradcam to base64 for frontend
        gradcam_base64 = None
        if gradcam_img is not None:
            buffered = BytesIO()
            gradcam_img.save(buffered, format="PNG")
            gradcam_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Parse model scores from details
        scores = parse_model_scores(details)
        
        return JSONResponse({
            "success": True,
            "risk_percentage": risk_pct,
            "risk_label": risk_label,
            "verdict": verdict,
            "details": details,
            "model_scores": scores,
            "gradcam": gradcam_base64,
            "mode": mode
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@fastapi_app.post("/api/analyze/video")
async def analyze_video_api(
    file: UploadFile = File(...),
    fps: float = Form(6),
    aggregation: str = Form("weighted_avg")
):
    """Analyze video for deepfakes"""
    try:
        # Save video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        missing_advanced = (face_model is None and eff_model is None and dino_model is None)
        mode = "ensemble"
        message = ""

        if missing_advanced:
            if corefakenet_model is not None:
                mode = "fast_mode"
                message = "Limited analysis (falling back to CorefakeNet Fast Mode)"
                # Fast mode execution
                from core_models.corefakenet import FastVideoProcessor
                fast_vp = FastVideoProcessor(corefakenet_path, device=str(device))
                result = fast_vp.analyze(tmp_path, sampling_fps=fps, progress_callback=None)
                
                os.unlink(tmp_path)
                
                if "error" in result:
                    return JSONResponse({
                        "success": False,
                        "error": result["error"]
                    }, status_code=200)
                
                risk_pct = result["final_risk"] * 100
                conf_pct = result["confidence"] * 100
                risk_label = f"Verdict: {result['prediction']} | AI Risk: {risk_pct:.1f}% | Confidence: {conf_pct:.1f}%"
                
                details_lines = [
                    f"Used CorefakeNet Fast Mode (fallback due to missing models).",
                    f"Elapsed: {result['elapsed_seconds']}s",
                    "--- Model Scores ---"
                ]
                for key, val in result.get('model_scores', {}).items():
                    details_lines.append(f"{key}: {val:.4f}")
                
                return JSONResponse({
                    "success": True,
                    "mode": mode,
                    "message": message,
                    "risk_percentage": risk_pct,
                    "risk_label": risk_label,
                    "verdict": "HIGH RISK" if risk_pct > 70 else "MEDIUM RISK" if risk_pct > 40 else "LOW RISK",
                    "details": "\n".join(details_lines),
                    "frame_details": "",
                    "gradcam_video": None,
                    "fps_analyzed": fps,
                    "aggregation": aggregation
                })
            else:
                mode = "fallback"
                message = "Limited analysis (only ViT available)"
        
        # Call existing video analysis (Ensemble / Fallback to ViT)
        risk_label, details, frame_details, gradcam_video_path = analyze_video_ui(
            tmp_path, fps, aggregation, progress=None
        )
        
        # Parse risk percentage
        import re
        match = re.search(r'Risk:\s*([\d.]+)%', risk_label)
        risk_pct = float(match.group(1)) if match else 50.0
        
        # Convert gradcam video to base64 if exists
        gradcam_base64 = None
        if gradcam_video_path and os.path.exists(gradcam_video_path):
            with open(gradcam_video_path, "rb") as f:
                import base64
                gradcam_base64 = base64.b64encode(f.read()).decode()
            os.unlink(gradcam_video_path)
        
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return JSONResponse({
            "success": True,
            "mode": mode,
            "message": message,
            "risk_percentage": risk_pct,
            "risk_label": risk_label,
            "verdict": "HIGH RISK" if risk_pct > 70 else "MEDIUM RISK" if risk_pct > 40 else "LOW RISK",
            "details": details,
            "frame_details": frame_details,
            "gradcam_video": gradcam_base64,
            "fps_analyzed": fps,
            "aggregation": aggregation
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=200)


@fastapi_app.post("/api/analyze/audio")
async def analyze_audio_api(
    file: UploadFile = File(...)
):
    """Analyze audio for deepfakes"""
    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Call existing audio analysis
        risk_label, details, verdict = analyze_audio_ui(tmp_path, progress=None)
        
        # Parse authenticity
        match = re.search(r'Authenticity:\s*([\d.]+)%', risk_label)
        auth_pct = float(match.group(1)) if match else 50.0
        risk_pct = 100 - auth_pct
        
        # Cleanup
        os.unlink(tmp_path)
        
        return JSONResponse({
            "success": True,
            "risk_percentage": risk_pct,
            "authenticity_percentage": auth_pct,
            "risk_label": risk_label,
            "verdict": verdict,
            "details": details
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@fastapi_app.post("/api/analyze/multimodal")
async def analyze_multimodal_api(
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """Multimodal analysis combining image, video, and audio"""
    try:
        # Process image if provided
        image_pil = None
        if image:
            contents = await image.read()
            image_pil = Image.open(BytesIO(contents)).convert("RGB")
        
        # Process video if provided
        video_path = None
        if video:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(await video.read())
                video_path = tmp.name
        
        # Process audio if provided
        audio_path = None
        if audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(await audio.read())
                audio_path = tmp.name
        
        # Call existing multimodal analysis
        risk_label, output_json, verdict = analyze_multimodal(
            image_pil, video_path, audio_path, progress=None
        )
        
        # Parse result
        data = json.loads(output_json)
        
        # Cleanup temp files
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        
        return JSONResponse({
            "success": True,
            "data": data,
            "risk_label": risk_label,
            "verdict": verdict
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# ========== REACT FRONTEND INTEGRATION ==========
# To use React, build your React app and place the build folder in 'frontend/build'
REACT_BUILD_PATH = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.exists(REACT_BUILD_PATH):
    fastapi_app.mount("/", StaticFiles(directory=REACT_BUILD_PATH, html=True), name="react")


# ========== MAIN - Choose between Gradio and FastAPI+React ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ProofyX Deepfake Detection")
    parser.add_argument("--ui", choices=["gradio", "react"], default="gradio",
                       help="Choose UI framework (gradio or react)")
    parser.add_argument("--port", type=int, default=7861,
                       help="Port to run the server on")
    
    args = parser.parse_args()
    
    if args.ui == "react":
        # Run FastAPI with React frontend
        import uvicorn
        print(f"Starting ProofyX React UI on http://127.0.0.1:{args.port}")
        print(f"API available at http://127.0.0.1:{args.port}/api")
        uvicorn.run(fastapi_app, host="127.0.0.1", port=args.port)
    else:
        # Run original Gradio UI
        LOGO_PATH = os.path.join(ROOT_DIR, "assets", "logo.jpeg")
        
        # Force dark mode JS injected at launch
        FORCE_DARK_JS = """
        () => {
            document.documentElement.classList.add('dark');
            document.body.classList.add('dark');
            document.body.style.backgroundColor = '#0A0E1A';
        }
        """
        
        CUSTOM_CSS = """
        /* ===== CSS Variables ===== */
        :root {
            --bg-primary: #0A0E1A;
            --bg-secondary: #0F1629;
            --bg-card: rgba(255,255,255,0.04);
            --bg-card-hover: rgba(255,255,255,0.07);
            --border-subtle: rgba(255,255,255,0.08);
            --border-glow: rgba(0,240,255,0.15);
            --accent-cyan: #00F0FF;
            --accent-violet: #A855F7;
            --accent-pink: #EC4899;
            --accent-green: #10B981;
            --accent-amber: #F59E0B;
            --text-primary: #E2E8F0;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --risk-low: #10B981;
            --risk-medium: #F59E0B;
            --risk-high: #EC4899;
            --glow-cyan: 0 0 20px rgba(0,240,255,0.15);
            --glow-violet: 0 0 20px rgba(168,85,247,0.15);
        }
        
        body, .gradio-container, .dark {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        .gradio-container {
            max-width: 1400px !important;
            margin: auto;
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }
        
        footer { display: none !important; }
        
        .tab-nav {
            background: rgba(255,255,255,0.02) !important;
            border-radius: 12px !important;
            padding: 4px !important;
            border: 1px solid var(--border-subtle) !important;
        }
        .tab-nav button {
            font-weight: 600 !important;
            color: var(--text-secondary) !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            background: transparent !important;
        }
        .tab-nav button.selected {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
            color: #fff !important;
        }
        
        .analyze-btn, button.primary {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
            color: #fff !important;
            border-radius: 12px !important;
            padding: 12px 28px !important;
            font-weight: 700 !important;
        }
        
        .proofyx-header {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 20px 0;
        }
        .proofyx-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 18px;
            border-radius: 10px;
            background: rgba(0,240,255,0.06);
            border: 1px solid rgba(0,240,255,0.15);
            color: var(--accent-cyan);
            margin-bottom: 16px;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse-glow 2s infinite;
        }
        
        @keyframes pulse-glow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        """
        
        def get_status_html():
            models_str = ", ".join(loaded_models) if loaded_models else "None"
            count = len(loaded_models)
            fast_mode = " | Fast Mode ready" if corefakenet_model is not None else ""
            return f'<div class="status-badge"><span class="status-dot"></span>{count} models active: {models_str}{fast_mode}</div>'
        
        def generate_gauge_html(risk_pct, label="Risk Score"):
            risk_pct = max(0, min(100, risk_pct))
            radius = 80
            circumference = 2 * 3.14159 * radius
            offset = circumference * (1 - risk_pct / 100)
            
            if risk_pct > 70:
                color = "#EC4899"
                glow = "rgba(236,72,153,0.4)"
            elif risk_pct > 40:
                color = "#F59E0B"
                glow = "rgba(245,158,11,0.4)"
            else:
                color = "#10B981"
                glow = "rgba(16,185,129,0.4)"
            
            return f"""
            <div style="display:flex;flex-direction:column;align-items:center;padding:20px 0;">
                <svg width="200" height="200" viewBox="0 0 200 200">
                    <circle cx="100" cy="100" r="{radius}" fill="none"
                            stroke="rgba(255,255,255,0.06)" stroke-width="12"/>
                    <circle cx="100" cy="100" r="{radius}" fill="none"
                            stroke="{color}" stroke-width="12"
                            stroke-linecap="round"
                            stroke-dasharray="{circumference}"
                            stroke-dashoffset="{offset}"
                            transform="rotate(-90 100 100)"
                            style="filter:drop-shadow(0 0 8px {glow});transition:stroke-dashoffset 0.8s ease;"/>
                    <text x="100" y="92" text-anchor="middle" fill="{color}"
                          font-size="36" font-weight="800" font-family="Inter,sans-serif">
                        {risk_pct:.0f}%
                    </text>
                    <text x="100" y="116" text-anchor="middle" fill="#94A3B8"
                          font-size="12" font-weight="500" font-family="Inter,sans-serif">
                        {label}
                    </text>
                </svg>
            </div>
            """
        
        def generate_score_bars_html(scores_dict):
            if not scores_dict:
                return '<div style="color:#64748B;text-align:center;padding:16px;">No scores available</div>'
            
            bars_html = ""
            for name, value in scores_dict.items():
                pct = max(0, min(100, value * 100))
                if pct > 70:
                    color = "#EC4899"
                    glow = "rgba(236,72,153,0.3)"
                elif pct > 40:
                    color = "#F59E0B"
                    glow = "rgba(245,158,11,0.3)"
                else:
                    color = "#10B981"
                    glow = "rgba(16,185,129,0.3)"
                
                bars_html += f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#CBD5E1;font-size:0.8rem;">{name}</span>
                        <span style="color:{color};font-size:0.8rem;font-weight:700;">{pct:.1f}%</span>
                    </div>
                    <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">
                        <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;
                                    box-shadow:0 0 8px {glow};transition:width 0.8s ease;"></div>
                    </div>
                </div>
                """
            return f'<div style="padding:12px 0;">{bars_html}</div>'
        
        def generate_verdict_html(verdict_str):
            if not verdict_str:
                return ""
            upper = verdict_str.upper()
            if "CRITICAL" in upper or "HIGH" in upper:
                bg = "rgba(236,72,153,0.1)"
                border = "rgba(236,72,153,0.3)"
                color = "#EC4899"
                icon = "&#9888;"
            elif "MEDIUM" in upper:
                bg = "rgba(245,158,11,0.1)"
                border = "rgba(245,158,11,0.3)"
                color = "#F59E0B"
                icon = "&#9888;"
            else:
                bg = "rgba(16,185,129,0.1)"
                border = "rgba(16,185,129,0.3)"
                color = "#10B981"
                icon = "&#10003;"
            
            return f"""
            <div style="padding:14px 18px;border-radius:12px;background:{bg};border:1px solid {border};margin-top:8px;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="font-size:1.1rem;color:{color};">{icon}</span>
                    <span style="font-weight:700;font-size:0.9rem;color:{color};">VERDICT</span>
                </div>
                <div style="color:#CBD5E1;font-size:0.82rem;line-height:1.5;">{verdict_str}</div>
            </div>
            """
        
        def analyze_image_ui_wrapper(image, mode):
            if image is None:
                empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload an image to analyze</div>'
                return empty, empty, "", "", None, None
            
            risk_label, details, verdict, gradcam_img = analyze_image_routed(image, mode)
            
            try:
                risk_pct = float(risk_label.split(":")[1].strip().replace("%", ""))
            except (IndexError, ValueError):
                risk_pct = 0.0
            
            gauge_html = generate_gauge_html(risk_pct, "AI Risk")
            scores = parse_model_scores(details)
            scores_html = generate_score_bars_html(scores)
            verdict_html = generate_verdict_html(verdict)
            
            return gauge_html, scores_html, verdict_html, details, gradcam_img, image
        
        def analyze_video_ui_wrapper(video, fps, aggregation, progress=gr.Progress()):
            if video is None:
                empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload a video to analyze</div>'
                return empty, empty, "", "", None
            
            risk_label, details, frame_details, gradcam_video = analyze_video_ui(video, fps, aggregation, progress)
            
            match = re.search(r'Risk:\s*([\d.]+)%', risk_label)
            risk_pct = float(match.group(1)) if match else 50.0
            
            gauge_html = generate_gauge_html(risk_pct, "Video Risk")
            summary_html = f'<div style="padding:12px 0;"><div style="color:#94A3B8;font-size:0.82rem;white-space:pre-wrap;">{risk_label}</div></div>'
            verdict_html = generate_verdict_html("HIGH RISK" if risk_pct > 70 else "MEDIUM RISK" if risk_pct > 40 else "LOW RISK")
            
            return gauge_html, summary_html, verdict_html, f"{details}\n\n{frame_details}", gradcam_video
        
        def analyze_audio_ui_wrapper(audio, progress=gr.Progress()):
            if audio is None:
                empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload audio to analyze</div>'
                return empty, empty, ""
            
            risk_label, details, verdict = analyze_audio_ui(audio, progress)
            
            match = re.search(r'Authenticity:\s*([\d.]+)%', risk_label)
            auth_pct = float(match.group(1)) if match else 50.0
            risk_pct = 100 - auth_pct
            
            gauge_html = generate_gauge_html(risk_pct, "Audio Risk")
            details_html = f'<div style="padding:12px 0;color:#CBD5E1;font-size:0.82rem;white-space:pre-wrap;">{details}</div>'
            verdict_html = generate_verdict_html(verdict)
            
            return gauge_html, details_html, verdict_html
        
        def analyze_multimodal_wrapper(image, video, audio, progress=gr.Progress()):
            if image is None and video is None and audio is None:
                empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload media to analyze</div>'
                return empty, empty, "", ""
            
            risk_label, output_json, verdict = analyze_multimodal(image, video, audio, progress)
            
            match = re.search(r'Risk:\s*([\d.]+)%', risk_label)
            risk_pct = float(match.group(1)) if match else 50.0
            
            gauge_html = generate_gauge_html(risk_pct, "Fused Risk")
            
            try:
                data = json.loads(output_json)
                mod_scores = data.get("modality_scores", {})
                bars = {mod.capitalize(): val / 100.0 for mod, val in mod_scores.items() if val is not None}
                bars_html = generate_score_bars_html(bars)
            except (json.JSONDecodeError, AttributeError):
                bars_html = '<div style="color:#64748B;">No modality data</div>'
            
            verdict_html = generate_verdict_html(verdict)
            
            return gauge_html, bars_html, verdict_html, output_json
        
        proofyx_theme = gr.themes.Base(
            primary_hue=gr.themes.Color(
                c50="#ecfeff", c100="#cffafe", c200="#a5f3fc", c300="#67e8f9",
                c400="#22d3ee", c500="#00F0FF", c600="#00d4e0", c700="#00a8b8",
                c800="#007a8a", c900="#005c68", c950="#003d45",
            ),
            secondary_hue=gr.themes.Color(
                c50="#faf5ff", c100="#f3e8ff", c200="#e9d5ff", c300="#d8b4fe",
                c400="#c084fc", c500="#A855F7", c600="#9333ea", c700="#7e22ce",
                c800="#6b21a8", c900="#581c87", c950="#3b0764",
            ),
            neutral_hue=gr.themes.Color(
                c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1",
                c400="#94a3b8", c500="#64748b", c600="#475569", c700="#334155",
                c800="#1e293b", c900="#0f172a", c950="#0A0E1A",
            ),
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0A0E1A",
            body_background_fill_dark="#0A0E1A",
            block_background_fill="rgba(255,255,255,0.04)",
            block_background_fill_dark="rgba(255,255,255,0.04)",
            block_border_width="1px",
            block_border_color="rgba(255,255,255,0.08)",
            block_border_color_dark="rgba(255,255,255,0.08)",
            block_radius="16px",
            button_primary_background_fill="linear-gradient(135deg, #00F0FF 0%, #A855F7 100%)",
            button_primary_background_fill_dark="linear-gradient(135deg, #00F0FF 0%, #A855F7 100%)",
            button_primary_text_color="white",
            input_border_color="rgba(255,255,255,0.08)",
            input_border_color_dark="rgba(255,255,255,0.08)",
            input_background_fill="rgba(255,255,255,0.04)",
            input_background_fill_dark="rgba(255,255,255,0.04)",
            input_radius="10px",
            body_text_color="#E2E8F0",
            body_text_color_dark="#E2E8F0",
            body_text_color_subdued="#94A3B8",
            body_text_color_subdued_dark="#94A3B8",
        )
        
        with gr.Blocks(title="ProofyX") as demo:
            gr.HTML(f"""
            <div class="proofyx-header">
                <img src="file/assets/logo.jpeg" alt="ProofyX Logo" style="width:64px;height:64px;border-radius:14px;" />
                <div>
                    <h1>PROOFYX</h1>
                    <p>AI Forensic Analysis &mdash; Deepfake &amp; Manipulation Detection</p>
                </div>
            </div>
            """)
            gr.HTML(get_status_html())
            
            with gr.Tabs():
                with gr.TabItem("Image"):
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["panel-left"]):
                            input_image = gr.Image(type="pil", label="Upload Image", elem_classes=["upload-area"])
                            analysis_mode = gr.Radio(
                                choices=["Full Ensemble (7 models)", "Fast Mode (CorefakeNet)"],
                                value="Full Ensemble (7 models)" if corefakenet_model is None else "Fast Mode (CorefakeNet)",
                                label="Analysis Mode"
                            )
                            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
                        
                        with gr.Column(scale=2, elem_classes=["panel-center"]):
                            img_display = gr.Image(label="Preview", type="pil")
                            heatmap_toggle = gr.Checkbox(label="Show Artifact Heatmap", value=True)
                        
                        with gr.Column(scale=1, elem_classes=["panel-right"]):
                            img_gauge = gr.HTML(value='<div style="text-align:center;padding:24px;">Awaiting analysis...</div>')
                            img_scores = gr.HTML()
                            img_verdict = gr.HTML()
                            with gr.Accordion("Raw Details", open=False):
                                img_details = gr.Textbox(lines=12, interactive=False, show_label=False)
                    
                    img_gradcam_state = gr.State(value=None)
                    img_original_state = gr.State(value=None)
                    
                    def _run_image(image, mode):
                        return analyze_image_ui_wrapper(image, mode)
                    
                    def _toggle_heatmap(show_heatmap, gradcam_img, original_img):
                        if show_heatmap and gradcam_img is not None:
                            return gradcam_img
                        return original_img
                    
                    analyze_btn.click(
                        fn=_run_image,
                        inputs=[input_image, analysis_mode],
                        outputs=[img_gauge, img_scores, img_verdict, img_details, img_gradcam_state, img_original_state],
                    ).then(
                        fn=_toggle_heatmap,
                        inputs=[heatmap_toggle, img_gradcam_state, img_original_state],
                        outputs=[img_display],
                    )
                    
                    input_image.change(
                        fn=_run_image,
                        inputs=[input_image, analysis_mode],
                        outputs=[img_gauge, img_scores, img_verdict, img_details, img_gradcam_state, img_original_state],
                    ).then(
                        fn=_toggle_heatmap,
                        inputs=[heatmap_toggle, img_gradcam_state, img_original_state],
                        outputs=[img_display],
                    )
                    
                    heatmap_toggle.change(
                        fn=_toggle_heatmap,
                        inputs=[heatmap_toggle, img_gradcam_state, img_original_state],
                        outputs=[img_display],
                    )
                
                with gr.TabItem("Video"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            input_video = gr.Video(label="Upload Video")
                            fps_slider = gr.Slider(minimum=0.5, maximum=10, value=6, step=0.5, label="Sampling FPS")
                            agg_method = gr.Dropdown(choices=["weighted_avg", "majority", "average", "max"], value="weighted_avg", label="Aggregation")
                            video_btn = gr.Button("Analyze Video", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            gradcam_video_output = gr.Video(label="GradCAM Detection Output")
                        
                        with gr.Column(scale=1):
                            vid_gauge = gr.HTML(value='<div style="text-align:center;padding:24px;">Awaiting analysis...</div>')
                            vid_summary = gr.HTML()
                            vid_verdict = gr.HTML()
                            with gr.Accordion("Frame Details", open=False):
                                vid_frames = gr.Textbox(lines=15, interactive=False, show_label=False)
                    
                    video_btn.click(
                        fn=analyze_video_ui_wrapper,
                        inputs=[input_video, fps_slider, agg_method],
                        outputs=[vid_gauge, vid_summary, vid_verdict, vid_frames, gradcam_video_output],
                    )
                
                with gr.TabItem("Audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            input_audio = gr.Audio(type="filepath", label="Upload Audio")
                            audio_btn = gr.Button("Analyze Audio", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            audio_center = gr.HTML(value='<div style="text-align:center;padding:60px 24px;">Upload an audio file and click Analyze</div>')
                        
                        with gr.Column(scale=1):
                            aud_gauge = gr.HTML(value='<div style="text-align:center;padding:24px;">Awaiting analysis...</div>')
                            aud_details = gr.HTML()
                            aud_verdict = gr.HTML()
                    
                    audio_btn.click(
                        fn=analyze_audio_ui_wrapper,
                        inputs=[input_audio],
                        outputs=[aud_gauge, aud_details, aud_verdict],
                    )
                
                with gr.TabItem("Multimodal"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            mm_image = gr.Image(type="pil", label="Image (optional)")
                            mm_video = gr.Video(label="Video (optional)")
                            mm_audio = gr.Audio(type="filepath", label="Audio (optional)")
                            mm_btn = gr.Button("Analyze All", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            mm_center = gr.HTML(value='<div style="text-align:center;padding:60px 24px;">Upload one or more media types for cross-modal fusion</div>')
                        
                        with gr.Column(scale=1):
                            mm_gauge = gr.HTML(value='<div style="text-align:center;padding:24px;">Awaiting analysis...</div>')
                            mm_bars = gr.HTML()
                            mm_verdict_html = gr.HTML()
                            with gr.Accordion("Raw JSON", open=False):
                                mm_json = gr.Textbox(lines=12, interactive=False, show_label=False)
                    
                    mm_btn.click(
                        fn=analyze_multimodal_wrapper,
                        inputs=[mm_image, mm_video, mm_audio],
                        outputs=[mm_gauge, mm_bars, mm_verdict_html, mm_json],
                    )
            
            gr.HTML("""
            <div class="proofyx-footer" style="text-align:center;padding:20px 0;color:#64748B;font-size:0.78rem;border-top:1px solid rgba(255,255,255,0.08);margin-top:20px;">
                <span>&#9670;</span> Face-aligned input &bull; Multi-model Grad-CAM &bull;
                Learned Fusion MLP &bull; ViT + EfficientNet-B4 + Forensic + Frequency CNN
                &bull; <span>CorefakeNet</span> Fast Mode &bull; Audio CNN &bull; Multimodal Fusion
            </div>
            """)
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            theme=proofyx_theme,
            css=CUSTOM_CSS,
            js=FORCE_DARK_JS,
            allowed_paths=[os.path.join(ROOT_DIR, "assets")],
        )