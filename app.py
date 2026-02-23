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
# With auxiliary models (face, dino, frequency) contributing smaller shares
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

    progress(0, desc="Starting video analysis...")

    def progress_callback(current, total, message):
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
        progress(0.9, desc="Generating face-aligned GradCAM video...")
        gradcam_video_path = _generate_gradcam_video(video, fps)

    return risk_label, details, frame_details, gradcam_video_path


def _generate_gradcam_video(video_path, fps):
    """Create side-by-side video with face-aligned GradCAM overlays."""
    info = get_video_info(video_path)
    if info is None:
        return None

    import imageio

    frame_w, frame_h = 400, 400
    gap = 6
    canvas_w = frame_w * 2 + gap
    label_h = 40
    canvas_h = frame_h + label_h

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()

    out_fps = max(1, int(fps))
    writer = imageio.get_writer(
        tmp_out.name, fps=out_fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
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

    writer.close()
    return tmp_out.name


# -------- Audio Prediction --------
def analyze_audio_ui(audio, progress=gr.Progress()):
    if audio is None:
        return "", "", ""

    progress(0, desc="Starting audio analysis...")

    def progress_callback(current, total, message):
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
        progress(0.3, desc="Analyzing video...")
        vid_result = video_analyzer_instance.analyze(video_path=video, fps=4, aggregation="weighted_avg")
        if "error" not in vid_result:
            vid_score = vid_result["avg_risk"]
            modality_scores["video"] = round(vid_score * 100, 1)
            results["video"] = {"score": vid_score, "prediction": vid_result["prediction"]}

    if audio is not None:
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

    import json
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


# -------- Gradio UI --------
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

/* ===== Global Reset ===== */
body, .gradio-container, .dark {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* Grid pattern background */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,240,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,240,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

footer { display: none !important; }

/* ===== Glassmorphism Cards ===== */
.glass-card, .block, .form, .panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}
.block:hover, .form:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--border-glow) !important;
}

/* ===== Tab Pills ===== */
.tab-nav {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border-subtle) !important;
    gap: 4px !important;
}
.tab-nav button {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.3s ease !important;
}
.tab-nav button:hover {
    color: var(--accent-cyan) !important;
    background: rgba(0,240,255,0.05) !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
    color: #fff !important;
    box-shadow: 0 4px 15px rgba(0,240,255,0.3), 0 4px 15px rgba(168,85,247,0.2) !important;
    border: none !important;
}

/* ===== Buttons ===== */
.analyze-btn, button.primary {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
    box-shadow: 0 0 20px rgba(0,240,255,0.25), 0 0 40px rgba(168,85,247,0.15) !important;
    transition: all 0.3s ease !important;
    cursor: pointer;
}
.analyze-btn:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(0,240,255,0.4), 0 0 60px rgba(168,85,247,0.25) !important;
}

/* ===== Upload Areas ===== */
.upload-area .image-container,
.upload-area .video-container,
.upload-area .audio-container,
.upload-area [data-testid="image"],
.upload-area [data-testid="droparea"] {
    border: 2px dashed rgba(0,240,255,0.3) !important;
    border-radius: 14px !important;
    background: rgba(0,240,255,0.02) !important;
    transition: all 0.3s ease !important;
}
.upload-area:hover [data-testid="image"],
.upload-area:hover [data-testid="droparea"],
.upload-area:hover .image-container {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 20px rgba(0,240,255,0.15) !important;
}

/* ===== Input Elements ===== */
input, textarea, select, .wrap {
    background: rgba(255,255,255,0.04) !important;
    border-color: var(--border-subtle) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 10px rgba(0,240,255,0.15) !important;
}
label, .label-wrap span {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}
.accordion {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}

/* ===== Header ===== */
.proofyx-header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 20px 0;
}
.proofyx-header img {
    width: 64px; height: 64px;
    border-radius: 14px;
    box-shadow: 0 0 20px rgba(0,240,255,0.2);
}
.proofyx-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}
.proofyx-header p {
    margin: 2px 0 0 0;
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* ===== Status Badge ===== */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: 10px;
    background: rgba(0,240,255,0.06);
    border: 1px solid rgba(0,240,255,0.15);
    color: var(--accent-cyan);
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 16px;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse-glow 2s infinite;
}

/* ===== Mode Toggle (Radio) ===== */
.mode-toggle .wrap {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
}

/* ===== Heatmap Toggle ===== */
.heatmap-toggle {
    margin-top: 8px;
}

/* ===== 3-Column Layout ===== */
.panel-left { min-width: 280px !important; }
.panel-center { min-width: 300px !important; }
.panel-right { min-width: 280px !important; }

/* ===== Result HTML Containers ===== */
.gauge-container, .scores-container, .verdict-container, .center-display {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* ===== Animations ===== */
@keyframes score-bar-fill {
    from { width: 0%; }
    to { width: var(--fill-width); }
}
@keyframes gauge-draw {
    from { stroke-dashoffset: var(--gauge-circumference); }
    to { stroke-dashoffset: var(--gauge-offset); }
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--accent-green); }
    50% { opacity: 0.5; box-shadow: 0 0 4px var(--accent-green); }
}
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ===== Progress Override ===== */
.progress-bar {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet)) !important;
}
.eta-bar {
    background: rgba(0,240,255,0.1) !important;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ===== Footer ===== */
.proofyx-footer {
    text-align: center;
    padding: 20px 0 10px 0;
    color: var(--text-muted);
    font-size: 0.78rem;
    border-top: 1px solid var(--border-subtle);
    margin-top: 20px;
}
.proofyx-footer span {
    color: var(--accent-cyan);
}

/* ===== Responsive ===== */
@media (max-width: 1024px) {
    .panel-left, .panel-center, .panel-right {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
}
"""


def get_status_html():
    models_str = ", ".join(loaded_models) if loaded_models else "None"
    count = len(loaded_models)
    fast_mode = " | Fast Mode ready" if corefakenet_model is not None else ""
    return (
        f'<div class="status-badge">'
        f'<span class="status-dot"></span>'
        f'{count} models active: {models_str}{fast_mode}'
        f'</div>'
    )


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
    block_shadow="none",
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


# -------- HTML Generator Functions --------

def generate_gauge_html(risk_pct, label="Risk Score"):
    """SVG circular gauge with animated arc stroke."""
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
    <div style="display:flex;flex-direction:column;align-items:center;padding:20px 0;
                animation:fade-in-up 0.5s ease-out;">
        <svg width="200" height="200" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="{radius}" fill="none"
                    stroke="rgba(255,255,255,0.06)" stroke-width="12"/>
            <circle cx="100" cy="100" r="{radius}" fill="none"
                    stroke="{color}" stroke-width="12"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"
                    transform="rotate(-90 100 100)"
                    style="--gauge-circumference:{circumference};--gauge-offset:{offset};
                           animation:gauge-draw 1.2s ease-out;
                           filter:drop-shadow(0 0 8px {glow});
                           transition:stroke-dashoffset 0.8s ease;"/>
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
    """Animated horizontal score bars for per-model breakdown."""
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
        <div style="margin-bottom:10px;animation:fade-in-up 0.4s ease-out;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#CBD5E1;font-size:0.8rem;font-weight:500;">{name}</span>
                <span style="color:{color};font-size:0.8rem;font-weight:700;">{pct:.1f}%</span>
            </div>
            <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;
                            box-shadow:0 0 8px {glow};
                            --fill-width:{pct}%;
                            animation:score-bar-fill 0.8s ease-out;"></div>
            </div>
        </div>
        """

    return f'<div style="padding:12px 0;">{bars_html}</div>'


def generate_verdict_html(verdict_str):
    """Color-coded verdict badge card."""
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
    <div style="padding:14px 18px;border-radius:12px;
                background:{bg};border:1px solid {border};
                animation:fade-in-up 0.5s ease-out;margin-top:8px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="font-size:1.1rem;color:{color};">{icon}</span>
            <span style="font-weight:700;font-size:0.9rem;color:{color};">VERDICT</span>
        </div>
        <div style="color:#CBD5E1;font-size:0.82rem;line-height:1.5;">{verdict_str}</div>
    </div>
    """


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


# -------- UI Wrapper Functions --------

def analyze_image_ui_wrapper(image, mode):
    """Wraps analyze_image_routed, transforms output for 3-panel HTML UI."""
    if image is None:
        empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload an image to analyze</div>'
        return empty, empty, "", "", None, None

    risk_label, details, verdict, gradcam_img = analyze_image_routed(image, mode)

    # Parse risk percentage from label
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
    """Wraps analyze_video_ui for 3-panel HTML UI."""
    if video is None:
        empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload a video to analyze</div>'
        return empty, empty, "", "", None

    risk_label, details, frame_details, gradcam_video = analyze_video_ui(
        video, fps, aggregation, progress
    )

    # Parse risk percentage
    try:
        import re
        match = re.search(r'Risk:\s*([\d.]+)%', risk_label)
        risk_pct = float(match.group(1)) if match else 50.0
    except (ValueError, AttributeError):
        risk_pct = 50.0

    gauge_html = generate_gauge_html(risk_pct, "Video Risk")

    # Build summary HTML from details
    summary_html = f"""
    <div style="padding:12px 0;">
        <div style="color:#94A3B8;font-size:0.82rem;line-height:1.7;
                    white-space:pre-wrap;font-family:'JetBrains Mono',monospace;">{risk_label}</div>
    </div>
    """
    verdict_html = generate_verdict_html(
        "HIGH RISK" if risk_pct > 70 else "MEDIUM RISK" if risk_pct > 40 else "LOW RISK"
    )

    return gauge_html, summary_html, verdict_html, f"{details}\n\n{frame_details}", gradcam_video


def analyze_audio_ui_wrapper(audio, progress=gr.Progress()):
    """Wraps analyze_audio_ui for 3-panel HTML UI."""
    if audio is None:
        empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload audio to analyze</div>'
        return empty, empty, ""

    risk_label, details, verdict = analyze_audio_ui(audio, progress)

    # Parse risk
    try:
        import re
        match = re.search(r'Authenticity:\s*([\d.]+)%', risk_label)
        auth_pct = float(match.group(1)) if match else 50.0
        risk_pct = 100 - auth_pct
    except (ValueError, AttributeError):
        risk_pct = 50.0

    gauge_html = generate_gauge_html(risk_pct, "Audio Risk")

    details_html = f"""
    <div style="padding:12px 0;color:#CBD5E1;font-size:0.82rem;line-height:1.7;
                white-space:pre-wrap;font-family:'JetBrains Mono',monospace;">{details}</div>
    """
    verdict_html = generate_verdict_html(verdict)

    return gauge_html, details_html, verdict_html


def analyze_multimodal_wrapper(image, video, audio, progress=gr.Progress()):
    """Wraps analyze_multimodal for 3-panel HTML UI."""
    if image is None and video is None and audio is None:
        empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload media to analyze</div>'
        return empty, empty, "", ""

    risk_label, output_json, verdict = analyze_multimodal(image, video, audio, progress)

    # Parse risk
    try:
        import re
        match = re.search(r'Risk:\s*([\d.]+)%', risk_label)
        risk_pct = float(match.group(1)) if match else 50.0
    except (ValueError, AttributeError):
        risk_pct = 50.0

    gauge_html = generate_gauge_html(risk_pct, "Fused Risk")

    # Parse modality bars from JSON
    import json
    try:
        data = json.loads(output_json)
        mod_scores = data.get("modality_scores", {})
        bars = {}
        for mod, val in mod_scores.items():
            if val is not None:
                bars[mod.capitalize()] = val / 100.0
        bars_html = generate_score_bars_html(bars)
    except (json.JSONDecodeError, AttributeError):
        bars_html = '<div style="color:#64748B;">No modality data</div>'

    verdict_html = generate_verdict_html(verdict)

    return gauge_html, bars_html, verdict_html, output_json


with gr.Blocks(title="ProofyX") as demo:

    # ===== HEADER =====
    gr.HTML(f"""
    <div class="proofyx-header">
        <img src="file/assets/logo.jpeg" alt="ProofyX Logo" />
        <div>
            <h1>PROOFYX</h1>
            <p>AI Forensic Analysis &mdash; Deepfake &amp; Manipulation Detection</p>
        </div>
    </div>
    """)
    gr.HTML(get_status_html())

    with gr.Tabs():
        # ===== IMAGE TAB =====
        with gr.TabItem("Image"):
            with gr.Row():
                # --- LEFT PANEL: Input ---
                with gr.Column(scale=1, elem_classes=["panel-left"]):
                    input_image = gr.Image(
                        type="pil", label="Upload Image",
                        elem_classes=["upload-area"],
                    )
                    analysis_mode = gr.Radio(
                        choices=[
                            "Full Ensemble (7 models)",
                            "Fast Mode (CorefakeNet)",
                        ],
                        value="Full Ensemble (7 models)" if corefakenet_model is None
                              else "Fast Mode (CorefakeNet)",
                        label="Analysis Mode",
                        elem_classes=["mode-toggle"],
                    )
                    analyze_btn = gr.Button(
                        "Analyze Image", variant="primary", size="lg",
                        elem_classes=["analyze-btn"],
                    )

                # --- CENTER PANEL: Viewer ---
                with gr.Column(scale=2, elem_classes=["panel-center"]):
                    img_display = gr.Image(
                        label="Preview", type="pil",
                        elem_classes=["center-display"],
                    )
                    heatmap_toggle = gr.Checkbox(
                        label="Show Artifact Heatmap",
                        value=True,
                        elem_classes=["heatmap-toggle"],
                    )

                # --- RIGHT PANEL: Results ---
                with gr.Column(scale=1, elem_classes=["panel-right"]):
                    img_gauge = gr.HTML(
                        elem_classes=["gauge-container"],
                        value='<div style="color:#64748B;text-align:center;padding:24px;">Awaiting analysis...</div>',
                    )
                    img_scores = gr.HTML(
                        elem_classes=["scores-container"],
                    )
                    img_verdict = gr.HTML(
                        elem_classes=["verdict-container"],
                    )
                    with gr.Accordion("Raw Details", open=False):
                        img_details = gr.Textbox(
                            lines=12, interactive=False,
                            show_label=False,
                        )

            # Hidden state for GradCAM image
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
                outputs=[img_gauge, img_scores, img_verdict, img_details,
                         img_gradcam_state, img_original_state],
            ).then(
                fn=_toggle_heatmap,
                inputs=[heatmap_toggle, img_gradcam_state, img_original_state],
                outputs=[img_display],
            )

            input_image.change(
                fn=_run_image,
                inputs=[input_image, analysis_mode],
                outputs=[img_gauge, img_scores, img_verdict, img_details,
                         img_gradcam_state, img_original_state],
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

        # ===== VIDEO TAB =====
        with gr.TabItem("Video"):
            with gr.Row():
                # --- LEFT ---
                with gr.Column(scale=1, elem_classes=["panel-left"]):
                    input_video = gr.Video(
                        label="Upload Video",
                        elem_classes=["upload-area"],
                    )
                    fps_slider = gr.Slider(
                        minimum=0.5, maximum=10, value=6, step=0.5,
                        label="Sampling FPS",
                    )
                    agg_method = gr.Dropdown(
                        choices=["weighted_avg", "majority", "average", "max"],
                        value="weighted_avg",
                        label="Aggregation",
                    )
                    video_btn = gr.Button(
                        "Analyze Video", variant="primary", size="lg",
                        elem_classes=["analyze-btn"],
                    )

                # --- CENTER ---
                with gr.Column(scale=2, elem_classes=["panel-center"]):
                    gradcam_video_output = gr.Video(
                        label="GradCAM Detection Output",
                        elem_classes=["center-display"],
                    )

                # --- RIGHT ---
                with gr.Column(scale=1, elem_classes=["panel-right"]):
                    vid_gauge = gr.HTML(
                        elem_classes=["gauge-container"],
                        value='<div style="color:#64748B;text-align:center;padding:24px;">Awaiting analysis...</div>',
                    )
                    vid_summary = gr.HTML(
                        elem_classes=["scores-container"],
                    )
                    vid_verdict = gr.HTML(
                        elem_classes=["verdict-container"],
                    )
                    with gr.Accordion("Frame Details", open=False):
                        vid_frames = gr.Textbox(
                            lines=15, interactive=False,
                            show_label=False,
                        )

            video_btn.click(
                fn=analyze_video_ui_wrapper,
                inputs=[input_video, fps_slider, agg_method],
                outputs=[vid_gauge, vid_summary, vid_verdict, vid_frames,
                         gradcam_video_output],
            )

        # ===== AUDIO TAB =====
        with gr.TabItem("Audio"):
            with gr.Row():
                # --- LEFT ---
                with gr.Column(scale=1, elem_classes=["panel-left"]):
                    input_audio = gr.Audio(
                        type="filepath", label="Upload Audio",
                        elem_classes=["upload-area"],
                    )
                    audio_btn = gr.Button(
                        "Analyze Audio", variant="primary", size="lg",
                        elem_classes=["analyze-btn"],
                    )

                # --- CENTER ---
                with gr.Column(scale=2, elem_classes=["panel-center"]):
                    audio_center = gr.HTML(
                        value='<div style="color:#64748B;text-align:center;padding:60px 24px;">'
                              'Upload an audio file and click Analyze<br>'
                              '<span style="font-size:0.8rem;">Supported: WAV, MP3, FLAC, M4A, OGG, AAC, WMA</span></div>',
                    )

                # --- RIGHT ---
                with gr.Column(scale=1, elem_classes=["panel-right"]):
                    aud_gauge = gr.HTML(
                        elem_classes=["gauge-container"],
                        value='<div style="color:#64748B;text-align:center;padding:24px;">Awaiting analysis...</div>',
                    )
                    aud_details = gr.HTML(
                        elem_classes=["scores-container"],
                    )
                    aud_verdict = gr.HTML(
                        elem_classes=["verdict-container"],
                    )

            audio_btn.click(
                fn=analyze_audio_ui_wrapper,
                inputs=[input_audio],
                outputs=[aud_gauge, aud_details, aud_verdict],
            )

        # ===== MULTIMODAL TAB =====
        with gr.TabItem("Multimodal"):
            with gr.Row():
                # --- LEFT ---
                with gr.Column(scale=1, elem_classes=["panel-left"]):
                    mm_image = gr.Image(
                        type="pil", label="Image (optional)",
                        elem_classes=["upload-area"],
                    )
                    mm_video = gr.Video(
                        label="Video (optional)",
                        elem_classes=["upload-area"],
                    )
                    mm_audio = gr.Audio(
                        type="filepath", label="Audio (optional)",
                        elem_classes=["upload-area"],
                    )
                    mm_btn = gr.Button(
                        "Analyze All", variant="primary", size="lg",
                        elem_classes=["analyze-btn"],
                    )

                # --- CENTER ---
                with gr.Column(scale=2, elem_classes=["panel-center"]):
                    mm_center = gr.HTML(
                        value='<div style="color:#64748B;text-align:center;padding:60px 24px;">'
                              'Upload one or more media types for cross-modal fusion analysis</div>',
                    )

                # --- RIGHT ---
                with gr.Column(scale=1, elem_classes=["panel-right"]):
                    mm_gauge = gr.HTML(
                        elem_classes=["gauge-container"],
                        value='<div style="color:#64748B;text-align:center;padding:24px;">Awaiting analysis...</div>',
                    )
                    mm_bars = gr.HTML(
                        elem_classes=["scores-container"],
                    )
                    mm_verdict_html = gr.HTML(
                        elem_classes=["verdict-container"],
                    )
                    with gr.Accordion("Raw JSON", open=False):
                        mm_json = gr.Textbox(
                            lines=12, interactive=False,
                            show_label=False,
                        )

            mm_btn.click(
                fn=analyze_multimodal_wrapper,
                inputs=[mm_image, mm_video, mm_audio],
                outputs=[mm_gauge, mm_bars, mm_verdict_html, mm_json],
            )

    # ===== FOOTER =====
    gr.HTML("""
    <div class="proofyx-footer">
        <span>&#9670;</span> Face-aligned input &bull; Multi-model Grad-CAM &bull;
        Learned Fusion MLP &bull; ViT + EfficientNet-B4 + Forensic + Frequency CNN
        &bull; <span>CorefakeNet</span> Fast Mode &bull; Audio CNN &bull; Multimodal Fusion
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        theme=proofyx_theme,
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS,
        allowed_paths=[os.path.join(ROOT_DIR, "assets")],
    )
