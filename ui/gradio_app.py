"""
ProofyX Gradio Dashboard — Cybersecurity-themed UI with sidebar navigation.

Multi-page layout: Scan, Analysis (results), History, Settings.
Uses gr.Sidebar for navigation, gr.HTML for custom components.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any, Optional

import gradio as gr

from ui.theme import (
    CUSTOM_CSS, FORCE_DARK_JS, STARS_JS,
    CHARTJS_HEAD, RADAR_JS, TIMELINE_JS,
    create_theme,
)
from ui.components import (
    generate_gauge_html, generate_score_bars_html, generate_verdict_html,
    generate_system_header, generate_modules_panel, generate_agreement_html,
    generate_exif_html, generate_history_html, parse_model_scores,
)
from core.pipeline import (
    analyze_image, analyze_video, analyze_audio, analyze_multimodal,
    get_registry,
)
from core.metadata import extract_full_metadata, extract_exif
from core.reports import build_forensic_report, generate_pdf_report, generate_html_report
from db.history import AnalysisHistory

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(ROOT_DIR, "assets", "logo.jpeg")

history_db = AnalysisHistory()


def create_gradio_app() -> gr.Blocks:
    """Build and return the full Gradio Blocks app."""

    reg = get_registry()
    session_id = str(uuid.uuid4())[:8]
    status = reg.get_status()

    with gr.Blocks(
        title="ProofyX",
    ) as demo:

        # ===== STARS BACKGROUND =====
        gr.HTML(
            value="",
            elem_id="stars-bg",
            visible=True,
        )

        # ===== SYSTEM HEADER =====
        header_html = generate_system_header(
            loaded_models=status["loaded"],
            session_id=session_id,
            corefakenet_ready=status["corefakenet_ready"],
        )
        gr.HTML(header_html)

        # ===== MAIN LAYOUT WITH SIDEBAR =====
        with gr.Row():

            # ===== LEFT SIDEBAR =====
            with gr.Column(scale=0, min_width=220, elem_classes=["nav-sidebar"]):
                if os.path.exists(LOGO_PATH):
                    gr.HTML(f"""
                    <div style="text-align:center;padding:16px 0 8px 0;">
                        <img src="/assets/logo.jpeg" alt="ProofyX"
                             style="width:48px;height:48px;border-radius:12px;
                                    box-shadow:0 0 16px rgba(0,240,255,0.2);" />
                        <div style="margin-top:8px;font-weight:800;font-size:1.1rem;
                                    background:linear-gradient(135deg,#00F0FF,#A855F7);
                                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                            PROOFYX
                        </div>
                    </div>
                    """)

                scan_nav = gr.Button("LIVE SCAN", elem_classes=["nav-btn", "nav-btn-active"])
                history_nav = gr.Button("HISTORY", elem_classes=["nav-btn"])
                settings_nav = gr.Button("SETTINGS", elem_classes=["nav-btn"])

                gr.HTML("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:12px 0;'/>")

                # Detection modules panel
                modules_html = generate_modules_panel(status["loaded"])
                gr.HTML(modules_html)

            # ===== MAIN CONTENT AREA =====
            with gr.Column(scale=4):

                # ──── PAGE: SCAN ────
                scan_page = gr.Column(visible=True)
                with scan_page:
                    with gr.Tabs():
                        # ===== IMAGE TAB =====
                        with gr.TabItem("Image"):
                            with gr.Row():
                                # Input panel
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
                                        value="Fast Mode (CorefakeNet)" if status["corefakenet_ready"]
                                              else "Full Ensemble (7 models)",
                                        label="Analysis Mode",
                                        elem_classes=["mode-toggle"],
                                    )
                                    analyze_btn = gr.Button(
                                        "INITIALIZE SCAN",
                                        variant="primary", size="lg",
                                        elem_classes=["analyze-btn"],
                                    )

                                # Center viewer
                                with gr.Column(scale=2, elem_classes=["panel-center"]):
                                    img_display = gr.Image(
                                        label="Preview", type="pil",
                                        elem_classes=["center-display"],
                                    )
                                    heatmap_toggle = gr.Checkbox(
                                        label="Show Artifact Heatmap", value=True,
                                    )
                                    # Radar chart
                                    img_radar = gr.HTML(
                                        value="",
                                        elem_classes=["radar-container"],
                                    )

                                # Results panel
                                with gr.Column(scale=1, elem_classes=["panel-right"]):
                                    img_gauge = gr.HTML(
                                        value='<div style="color:#64748B;text-align:center;padding:24px;">Awaiting scan...</div>',
                                        elem_classes=["gauge-container"],
                                    )
                                    img_agreement = gr.HTML()
                                    img_scores = gr.HTML(elem_classes=["scores-container"])
                                    img_verdict = gr.HTML(elem_classes=["verdict-container"])
                                    img_exif = gr.HTML()
                                    with gr.Accordion("Raw Details", open=False):
                                        img_details = gr.Textbox(lines=12, interactive=False, show_label=False)
                                    download_report_btn = gr.Button(
                                        "Download Report", variant="secondary", visible=False,
                                    )
                                    report_file = gr.File(label="Report", visible=False)

                            # States
                            img_gradcam_state = gr.State(value=None)
                            img_original_state = gr.State(value=None)
                            img_result_state = gr.State(value=None)

                            def _run_image(image, mode):
                                if image is None:
                                    empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload an image to scan</div>'
                                    return empty, "", empty, "", "", "", None, None, None, ""

                                pipeline_mode = "fast" if "Fast" in mode else "ensemble"
                                result = analyze_image(image, mode=pipeline_mode)

                                if "error" in result and result["error"]:
                                    return result["error"], "", "", "", "", "", None, None, None, ""

                                risk_pct = result["risk_percent"]
                                gauge = generate_gauge_html(risk_pct, "AI Risk")
                                agreement = generate_agreement_html(result.get("model_agreement", ""))
                                scores = generate_score_bars_html(result.get("model_scores", {}))
                                verdict = generate_verdict_html(result.get("verdict", ""))

                                # EXIF metadata
                                metadata = extract_full_metadata(image)
                                exif_html = generate_exif_html(metadata)
                                result["metadata"] = metadata

                                # Details
                                details_lines = []
                                details_lines.append(f"Face Detected      : {'Yes' if result.get('face_detected') else 'No'}")
                                details_lines.append(f"Fusion Mode        : {result.get('fusion_mode', '')}")
                                details_lines.append(f"Models Used        : {result.get('models_used', 0)}")
                                details_lines.append(f"Processing Time    : {result.get('processing_time_ms', 0):.0f}ms")
                                details_lines.append("")
                                for name, val in result.get("model_scores", {}).items():
                                    details_lines.append(f"{name:20s}: {val:.4f}")
                                details_lines.append(f"\nFinal Risk Score   : {result.get('risk_score', 0):.4f}")
                                details = "\n".join(details_lines)

                                # Save to history
                                result["file_name"] = "uploaded_image"
                                history_db.save(result)

                                gradcam_img = result.get("gradcam_image")
                                original_img = result.get("original_image", image)

                                return (gauge, agreement, scores, verdict,
                                        exif_html, details,
                                        gradcam_img, original_img, result, "")

                            def _toggle_heatmap(show, gradcam, original):
                                if show and gradcam is not None:
                                    return gradcam
                                return original

                            def _generate_report(result_dict):
                                if result_dict is None:
                                    return gr.update(visible=False)
                                report = build_forensic_report(
                                    result_dict,
                                    file_name=result_dict.get("file_name", ""),
                                    image_pil=result_dict.get("original_image"),
                                    gradcam_pil=result_dict.get("gradcam_image"),
                                )
                                import tempfile
                                tmp = tempfile.NamedTemporaryFile(
                                    suffix=".pdf", delete=False, prefix="proofyx_report_"
                                )
                                tmp.close()
                                path = generate_pdf_report(report, tmp.name)
                                return gr.update(value=path, visible=True)

                            analyze_btn.click(
                                fn=_run_image,
                                inputs=[input_image, analysis_mode],
                                outputs=[img_gauge, img_agreement, img_scores, img_verdict,
                                         img_exif, img_details,
                                         img_gradcam_state, img_original_state, img_result_state,
                                         img_radar],
                            ).then(
                                fn=_toggle_heatmap,
                                inputs=[heatmap_toggle, img_gradcam_state, img_original_state],
                                outputs=[img_display],
                            ).then(
                                fn=lambda _: gr.update(visible=True),
                                inputs=[img_result_state],
                                outputs=[download_report_btn],
                            )

                            heatmap_toggle.change(
                                fn=_toggle_heatmap,
                                inputs=[heatmap_toggle, img_gradcam_state, img_original_state],
                                outputs=[img_display],
                            )

                            download_report_btn.click(
                                fn=_generate_report,
                                inputs=[img_result_state],
                                outputs=[report_file],
                            )

                        # ===== VIDEO TAB =====
                        with gr.TabItem("Video"):
                            with gr.Row():
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
                                        value="weighted_avg", label="Aggregation",
                                    )
                                    video_btn = gr.Button(
                                        "INITIALIZE SCAN", variant="primary", size="lg",
                                        elem_classes=["analyze-btn"],
                                    )

                                with gr.Column(scale=2, elem_classes=["panel-center"]):
                                    gradcam_video_output = gr.Video(
                                        label="GradCAM Detection Output",
                                        elem_classes=["center-display"],
                                    )
                                    vid_timeline = gr.HTML(
                                        value="",
                                        elem_classes=["timeline-container"],
                                    )

                                with gr.Column(scale=1, elem_classes=["panel-right"]):
                                    vid_gauge = gr.HTML(
                                        value='<div style="color:#64748B;text-align:center;padding:24px;">Awaiting scan...</div>',
                                        elem_classes=["gauge-container"],
                                    )
                                    vid_summary = gr.HTML(elem_classes=["scores-container"])
                                    vid_verdict = gr.HTML(elem_classes=["verdict-container"])
                                    with gr.Accordion("Frame Details", open=False):
                                        vid_frames = gr.Textbox(lines=15, interactive=False, show_label=False)

                            def _run_video(video, fps, aggregation, progress=gr.Progress()):
                                if video is None:
                                    empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload a video to scan</div>'
                                    return empty, empty, "", "", None, ""

                                def progress_cb(current, total, msg):
                                    progress(current / max(total, 1), desc=msg)

                                result = analyze_video(video, fps=fps, aggregation=aggregation,
                                                       progress_callback=progress_cb)

                                if "error" in result and result["error"]:
                                    return result["error"], "", "", "", None, ""

                                risk_pct = result["risk_percent"]
                                gauge = generate_gauge_html(risk_pct, "Video Risk")

                                summary = f"""
                                <div style="padding:12px 0;color:#94A3B8;font-size:0.82rem;line-height:1.7;">
                                    Frames: {result.get('total_frames_analyzed', 0)} |
                                    Fake: {result.get('fake_frames', 0)} |
                                    Real: {result.get('real_frames', 0)}<br>
                                    Processing: {result.get('processing_time_ms', 0):.0f}ms
                                </div>"""

                                verdict = generate_verdict_html(result.get("verdict", ""))

                                # Frame details
                                frame_lines = []
                                for fr in result.get("frame_results", [])[:50]:
                                    frame_lines.append(
                                        f"Frame {fr.get('frame_index', 0):4d} | "
                                        f"{fr.get('timestamp', 0):5.1f}s | "
                                        f"Risk: {fr.get('frame_risk', 0):.3f}"
                                    )
                                frames_text = "\n".join(frame_lines)

                                # Save
                                result["file_name"] = os.path.basename(video) if video else ""
                                history_db.save(result)

                                return gauge, summary, verdict, frames_text, None, ""

                            video_btn.click(
                                fn=_run_video,
                                inputs=[input_video, fps_slider, agg_method],
                                outputs=[vid_gauge, vid_summary, vid_verdict, vid_frames,
                                         gradcam_video_output, vid_timeline],
                            )

                        # ===== AUDIO TAB =====
                        with gr.TabItem("Audio"):
                            with gr.Row():
                                with gr.Column(scale=1, elem_classes=["panel-left"]):
                                    input_audio = gr.Audio(
                                        type="filepath", label="Upload Audio",
                                        elem_classes=["upload-area"],
                                    )
                                    audio_btn = gr.Button(
                                        "INITIALIZE SCAN", variant="primary", size="lg",
                                        elem_classes=["analyze-btn"],
                                    )

                                with gr.Column(scale=2, elem_classes=["panel-center"]):
                                    audio_center = gr.HTML(
                                        value='<div style="color:#64748B;text-align:center;padding:60px 24px;">'
                                              'Upload an audio file and click Initialize Scan<br>'
                                              '<span style="font-size:0.8rem;">Supported: WAV, MP3, FLAC, M4A, OGG, AAC, WMA</span></div>',
                                    )

                                with gr.Column(scale=1, elem_classes=["panel-right"]):
                                    aud_gauge = gr.HTML(
                                        value='<div style="color:#64748B;text-align:center;padding:24px;">Awaiting scan...</div>',
                                        elem_classes=["gauge-container"],
                                    )
                                    aud_details = gr.HTML(elem_classes=["scores-container"])
                                    aud_verdict = gr.HTML(elem_classes=["verdict-container"])

                            def _run_audio(audio, progress=gr.Progress()):
                                if audio is None:
                                    empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload audio to scan</div>'
                                    return empty, empty, ""

                                def progress_cb(current, total, msg):
                                    progress(current / max(total, 1), desc=msg)

                                result = analyze_audio(audio, progress_callback=progress_cb)

                                if "error" in result and result["error"]:
                                    return result["error"], "", ""

                                auth_score = result.get("authenticity_score", 50)
                                risk_pct = 100 - auth_score
                                gauge = generate_gauge_html(risk_pct, "Audio Risk")

                                details = f"""
                                <div style="padding:12px 0;color:#CBD5E1;font-size:0.82rem;line-height:1.7;">
                                    Duration: {result.get('duration_sec', 0)}s<br>
                                    Segments: {result.get('segments_analyzed', 0)}<br>
                                    Manipulation: {result.get('manipulation_type', 'None')}<br>
                                    Evidence: {', '.join(result.get('evidence', []))}<br>
                                    Processing: {result.get('processing_time_ms', 0):.0f}ms
                                </div>"""

                                verdict = generate_verdict_html(result.get("verdict", ""))

                                # Save
                                result["file_name"] = os.path.basename(audio) if audio else ""
                                history_db.save(result)

                                return gauge, details, verdict

                            audio_btn.click(
                                fn=_run_audio,
                                inputs=[input_audio],
                                outputs=[aud_gauge, aud_details, aud_verdict],
                            )

                        # ===== MULTIMODAL TAB =====
                        with gr.TabItem("Multimodal"):
                            with gr.Row():
                                with gr.Column(scale=1, elem_classes=["panel-left"]):
                                    mm_image = gr.Image(type="pil", label="Image (optional)", elem_classes=["upload-area"])
                                    mm_video = gr.Video(label="Video (optional)", elem_classes=["upload-area"])
                                    mm_audio = gr.Audio(type="filepath", label="Audio (optional)", elem_classes=["upload-area"])
                                    mm_btn = gr.Button(
                                        "ANALYZE ALL", variant="primary", size="lg",
                                        elem_classes=["analyze-btn"],
                                    )

                                with gr.Column(scale=2, elem_classes=["panel-center"]):
                                    mm_center = gr.HTML(
                                        value='<div style="color:#64748B;text-align:center;padding:60px 24px;">'
                                              'Upload one or more media types for cross-modal fusion analysis</div>',
                                    )

                                with gr.Column(scale=1, elem_classes=["panel-right"]):
                                    mm_gauge = gr.HTML(
                                        value='<div style="color:#64748B;text-align:center;padding:24px;">Awaiting scan...</div>',
                                        elem_classes=["gauge-container"],
                                    )
                                    mm_bars = gr.HTML(elem_classes=["scores-container"])
                                    mm_verdict = gr.HTML(elem_classes=["verdict-container"])
                                    with gr.Accordion("Raw JSON", open=False):
                                        mm_json = gr.Textbox(lines=12, interactive=False, show_label=False)

                            def _run_multimodal(image, video, audio, progress=gr.Progress()):
                                if image is None and video is None and audio is None:
                                    empty = '<div style="color:#64748B;text-align:center;padding:24px;">Upload media to scan</div>'
                                    return empty, empty, "", ""

                                result = analyze_multimodal(
                                    image=image, video_path=video, audio_path=audio,
                                )

                                if "error" in result and result["error"]:
                                    return result["error"], "", "", ""

                                risk_pct = result["risk_percent"]
                                gauge = generate_gauge_html(risk_pct, "Fused Risk")

                                mod_scores = result.get("modality_scores", {})
                                bars = {k.capitalize(): v for k, v in mod_scores.items() if v is not None}
                                bars_html = generate_score_bars_html(bars)

                                verdict = generate_verdict_html(result.get("verdict", ""))

                                import json
                                raw_json = json.dumps({
                                    "media_types": result.get("media_types", []),
                                    "risk_score": round(result.get("risk_score", 0), 4),
                                    "verdict": result.get("verdict", ""),
                                    "modality_scores": mod_scores,
                                    "fusion_weights": result.get("fusion_weights", {}),
                                    "explanation": result.get("explanation", ""),
                                }, indent=2)

                                return gauge, bars_html, verdict, raw_json

                            mm_btn.click(
                                fn=_run_multimodal,
                                inputs=[mm_image, mm_video, mm_audio],
                                outputs=[mm_gauge, mm_bars, mm_verdict, mm_json],
                            )

                # ──── PAGE: HISTORY ────
                history_page = gr.Column(visible=False)
                with history_page:
                    gr.HTML('<h2 style="color:#00F0FF;margin-bottom:16px;">Analysis History</h2>')
                    history_html = gr.HTML(
                        value=generate_history_html(history_db.get_recent(20)),
                    )
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    refresh_btn.click(
                        fn=lambda: generate_history_html(history_db.get_recent(20)),
                        outputs=[history_html],
                    )

                # ──── PAGE: SETTINGS ────
                settings_page = gr.Column(visible=False)
                with settings_page:
                    gr.HTML('<h2 style="color:#00F0FF;margin-bottom:16px;">Settings</h2>')
                    gr.HTML(f"""
                    <div style="padding:16px;background:rgba(255,255,255,0.03);
                                border:1px solid rgba(255,255,255,0.06);border-radius:12px;">
                        <div style="color:#94A3B8;font-size:0.82rem;line-height:2;">
                            <strong style="color:#E2E8F0;">Session ID:</strong> {session_id}<br>
                            <strong style="color:#E2E8F0;">Models Loaded:</strong> {len(status['loaded'])}<br>
                            <strong style="color:#E2E8F0;">Models Missing:</strong> {len(status['missing'])}<br>
                            <strong style="color:#E2E8F0;">CorefakeNet:</strong> {'Ready' if status['corefakenet_ready'] else 'Not loaded'}<br>
                            <strong style="color:#E2E8F0;">Device:</strong> CPU<br>
                            <strong style="color:#E2E8F0;">API:</strong> /api/v1 (REST), /docs (Swagger)<br>
                        </div>
                    </div>
                    """)

                # ──── NAVIGATION LOGIC ────
                def show_scan():
                    return (
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                    )

                def show_history():
                    html = generate_history_html(history_db.get_recent(20))
                    return (
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False),
                    )

                def show_settings():
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True),
                    )

                pages = [scan_page, history_page, settings_page]
                scan_nav.click(fn=show_scan, outputs=pages)
                history_nav.click(fn=show_history, outputs=pages)
                settings_nav.click(fn=show_settings, outputs=pages)

        # ===== FOOTER =====
        gr.HTML("""
        <div class="proofyx-footer">
            <span>&#9670;</span> Face-aligned input &bull; Multi-model GradCAM &bull;
            Learned Fusion MLP &bull; ViT + EfficientNet-B4 + Forensic + Frequency CNN
            &bull; <span>CorefakeNet</span> Fast Mode &bull; Audio CNN &bull; Multimodal Fusion
            &bull; <span>EXIF Forensics</span> &bull; PDF Reports &bull; REST API
        </div>
        """)

    return demo
