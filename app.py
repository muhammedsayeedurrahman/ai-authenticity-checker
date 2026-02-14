import sys
import os

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

from utils.explainability import explain_risk
from utils.gradcam import generate_gradcam_image, get_gradcam_for_face_model, create_heatmap_overlay, _preprocess
from pipeline.face_gate import face_present
from pipeline.video_analyzer import analyze_video, extract_frames, get_video_info

from core_models.dinov2_auth_model import DINOv2AuthModel
from core_models.efficientnet_auth_model import EfficientNetAuthModel
from core_models.face_deepfake_model import FaceDeepfakeModel

# -------- Config --------
MODELS_DIR = os.path.join(ROOT_DIR, "models")
WEIGHTS = {"dino": 0.4, "efficientnet": 0.35, "face": 0.25}
HIGH_CONFIDENCE_OVERRIDE = 0.9

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
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    loaded_models.append(name)
    return model


dino_model = try_load_model("DINOv2", DINOv2AuthModel, "dinov2_auth_model.pth")
eff_model = try_load_model("EfficientNet", EfficientNetAuthModel, "efficientnet_auth_model.pth")
face_model = try_load_model("Face Deepfake", FaceDeepfakeModel, "image_face_model.pth")

if loaded_models:
    print(f"Loaded models: {', '.join(loaded_models)}")
if missing_models:
    print(f"Missing model weights (train first): {', '.join(missing_models)}")


# -------- Image Prediction --------
def analyze_image(image):
    if image is None:
        return "", "", "", None

    # Save to temp file for face detection
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path)

    try:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

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

            has_face = face_present(tmp_path)
            if has_face and face_model is not None:
                real_prob = face_model(tensor).item()
                face_prob = 1.0 - real_prob
                active_models += 1

        if active_models == 0:
            status = "No trained models found. Please train models first."
            details = "Run the training scripts to generate model weights:\n"
            details += "  python training/train_dinov2.py\n"
            details += "  python training/train_efficientnet_auth.py\n"
            details += "  python training/train_face_image.py"
            return status, details, "", None

        # Weighted ensemble
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

        final_risk = weighted_sum / total_weight if total_weight > 0 else 0.0

        # High-confidence override
        max_prob = max(dino_prob, eff_prob, face_prob)
        if max_prob > HIGH_CONFIDENCE_OVERRIDE:
            final_risk = max(final_risk, max_prob)

        # Build verdict
        verdict = explain_risk(final_risk)
        risk_pct = final_risk * 100
        risk_label = f"AI-Generated Risk: {risk_pct:.1f}%"

        # Details
        details_lines = []
        details_lines.append(f"Face Detected      : {'Yes' if has_face else 'No'}")
        details_lines.append("")

        if dino_model is not None:
            details_lines.append(f"DINOv2 Score       : {dino_prob:.4f}")
        else:
            details_lines.append(f"DINOv2 Score       : N/A (model not trained)")

        if eff_model is not None:
            details_lines.append(f"EfficientNet Score : {eff_prob:.4f}")
        else:
            details_lines.append(f"EfficientNet Score : N/A (model not trained)")

        if has_face:
            if face_model is not None:
                details_lines.append(f"Face Fake Score    : {face_prob:.4f}")
            else:
                details_lines.append(f"Face Fake Score    : N/A (model not trained)")
        else:
            details_lines.append(f"Face Fake Score    : N/A (no face detected)")

        details_lines.append("")
        details_lines.append(f"Final Risk Score   : {final_risk:.4f}")
        details_lines.append(f"Active Models      : {active_models}")

        details = "\n".join(details_lines)

        # GradCAM side-by-side
        gradcam_img = generate_gradcam_image(image, face_model, device)

        return risk_label, details, verdict, gradcam_img

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# -------- Video Prediction with GradCAM --------
def analyze_video_ui(video, fps, aggregation, progress=gr.Progress()):
    if video is None:
        return "", "", "", None

    progress(0, desc="Starting video analysis...")

    def progress_callback(current, total, message):
        progress(current / max(total, 1), desc=message)

    result = analyze_video(
        video_path=video,
        dino_model=dino_model,
        eff_model=eff_model,
        face_model=face_model,
        device=device,
        fps=fps,
        aggregation=aggregation,
        progress_callback=progress_callback,
    )

    if "error" in result:
        return result["error"], "", "", None

    # Overall verdict
    risk_pct = result["avg_risk"] * 100
    prediction = result["prediction"]
    confidence = result["confidence"] * 100

    risk_label = f"Verdict: {prediction} | AI Risk: {risk_pct:.1f}% | Confidence: {confidence:.1f}%"

    # Summary details
    info = result["video_info"]
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

    details = "\n".join(lines)

    # Per-frame breakdown
    frame_lines = []
    frame_lines.append(f"{'Frame':<7} {'Time':>6} {'Risk':>7} {'Pred':>6} {'Face':>5} {'DINO':>7} {'EffNet':>7} {'FaceM':>7}")
    frame_lines.append("-" * 65)
    for fr in result["frame_results"]:
        frame_lines.append(
            f"  {fr['frame_index']:<5} {fr['timestamp']:>5.1f}s {fr['frame_risk']:>6.4f} "
            f"{fr['prediction']:>6} {'Y' if fr['has_face'] else 'N':>4}  "
            f"{fr['dino_prob']:>6.4f} {fr['eff_prob']:>7.4f} {fr['face_prob']:>6.4f}"
        )
    frame_details = "\n".join(frame_lines)

    # Generate GradCAM output video (side-by-side: original | heatmap)
    gradcam_video_path = None
    if face_model is not None:
        progress(0.9, desc="Generating GradCAM heatmap video...")
        gradcam_video_path = _generate_gradcam_video(video, fps)

    return risk_label, details, frame_details, gradcam_video_path


def _generate_gradcam_video(video_path, fps):
    """Create a side-by-side video: original frames | GradCAM heatmap overlay."""
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
    text_color = (0, 0, 0)  # Black text on white bg

    label_left = "Manipulated Video"
    label_right = "AI Auth Checker output"
    left_size = cv2.getTextSize(label_left, font, font_scale, thickness)[0]
    right_size = cv2.getTextSize(label_right, font, font_scale, thickness)[0]

    for frame_idx, pil_img, timestamp in extract_frames(video_path, fps=fps):
        # Generate heatmap
        tensor = _preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
        heatmap = get_gradcam_for_face_model(face_model, tensor)
        overlay_pil = create_heatmap_overlay(pil_img, heatmap, alpha=0.45)

        # Resize both
        original = np.array(pil_img.convert("RGB").resize((frame_w, frame_h)))
        overlay = np.array(overlay_pil.resize((frame_w, frame_h)))

        # Build canvas (white background)
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        # Labels (centered, black text)
        cv2.putText(canvas, label_left,
                     (frame_w // 2 - left_size[0] // 2, 28),
                     font, font_scale, text_color, thickness)
        cv2.putText(canvas, label_right,
                     (frame_w + gap + frame_w // 2 - right_size[0] // 2, 28),
                     font, font_scale, text_color, thickness)

        # Place frames
        canvas[label_h:, :frame_w] = original
        canvas[label_h:, frame_w + gap:] = overlay

        # imageio expects RGB directly
        writer.append_data(canvas)

    writer.close()
    return tmp_out.name


# -------- Gradio UI --------
def get_status_message():
    if not loaded_models:
        return (
            "No trained model weights found. "
            "You can still upload images, but predictions require trained models.\n\n"
            "Train models with:\n"
            "  python training/train_dinov2.py\n"
            "  python training/train_efficientnet_auth.py\n"
            "  python training/train_face_image.py"
        )
    msg = f"Running on: {device} | Models loaded: {', '.join(loaded_models)}"
    if missing_models:
        msg += f" | Missing: {', '.join(missing_models)}"
    return msg


with gr.Blocks() as demo:
    gr.Markdown("# AI Authenticity Checker")
    gr.Markdown(f"*{get_status_message()}*")

    with gr.Tabs():
        # ===== IMAGE TAB =====
        with gr.TabItem("Image Analysis"):
            gr.Markdown("Upload an image to check if it was AI-generated or real.")

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Image")
                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

                with gr.Column(scale=1):
                    risk_output = gr.Textbox(label="Risk Level", lines=1, interactive=False)
                    details_output = gr.Textbox(label="Model Scores", lines=10, interactive=False)
                    verdict_output = gr.Textbox(label="Verdict", lines=1, interactive=False)

            # GradCAM output below
            gr.Markdown("### Detection Heatmap")
            gradcam_output = gr.Image(label="Input vs Manipulation Heatmap", type="pil")

            analyze_btn.click(
                fn=analyze_image,
                inputs=[input_image],
                outputs=[risk_output, details_output, verdict_output, gradcam_output]
            )

            input_image.change(
                fn=analyze_image,
                inputs=[input_image],
                outputs=[risk_output, details_output, verdict_output, gradcam_output]
            )

        # ===== VIDEO TAB =====
        with gr.TabItem("Video Analysis"):
            gr.Markdown(
                "Upload a video to detect deepfakes. "
                "Frames are extracted and analyzed individually, then aggregated for a final verdict.\n\n"
                "*Inspired by [VeridisQuo](https://github.com/VeridisQuo-orga/VeridisQuo) — "
                "hybrid spatial + ensemble detection with frame-level aggregation.*"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="Upload Video")
                    with gr.Row():
                        fps_slider = gr.Slider(
                            minimum=0.5, maximum=5, value=1, step=0.5,
                            label="Sampling FPS",
                            info="Frames per second to extract (lower = faster)"
                        )
                        agg_method = gr.Dropdown(
                            choices=["majority", "average", "max"],
                            value="majority",
                            label="Aggregation",
                            info="How to combine per-frame predictions"
                        )
                    video_btn = gr.Button("Analyze Video", variant="primary", size="lg")

                with gr.Column(scale=1):
                    video_risk = gr.Textbox(label="Verdict", lines=1, interactive=False)
                    video_details = gr.Textbox(label="Video Summary", lines=10, interactive=False)
                    video_frames = gr.Textbox(
                        label="Per-Frame Breakdown", lines=15, interactive=False
                    )

            # GradCAM video output
            gr.Markdown("### GradCAM Detection Video")
            gr.Markdown("*Side-by-side: Original frames | Manipulation heatmap overlay*")
            gradcam_video_output = gr.Video(label="Manipulated Video vs Detection Output")

            video_btn.click(
                fn=analyze_video_ui,
                inputs=[input_video, fps_slider, agg_method],
                outputs=[video_risk, video_details, video_frames, gradcam_video_output]
            )

    gr.Markdown("---")
    gr.Markdown(
        "**Models:** DINOv2 (weight: 40%) | EfficientNet V2 (weight: 35%) | "
        "Face Deepfake ResNet50 (weight: 25%) | "
        "High-confidence override at >90%"
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, theme=gr.themes.Soft())
