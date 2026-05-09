"""
ProofyX Gradio Dashboard -- Minimal + Elegant + Deeptech UI.

Top navigation bar, 2-column scan layout, skeleton loading states,
smooth page transitions, clean footer.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import gradio as gr

from ui.theme import CUSTOM_CSS, FORCE_DARK_JS, CDN_HEAD, create_theme
from ui.components import (
    generate_gauge_html, generate_score_bars_html, generate_verdict_html,
    generate_top_nav, generate_modules_panel, generate_agreement_html,
    generate_exif_html, generate_history_html, parse_model_scores,
    generate_empty_state, generate_skeleton_gauge, generate_skeleton_bars,
    generate_skeleton_verdict,
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

    logo_url = "/assets/logo.jpeg" if os.path.exists(LOGO_PATH) else ""

    with gr.Blocks(
        title="ProofyX",
        theme=create_theme(),
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS,
        head=CDN_HEAD,
    ) as demo:

        # ===== TOP NAVIGATION BAR =====
        nav_html = generate_top_nav(
            loaded_models=status["loaded"],
            session_id=session_id,
            corefakenet_ready=status["corefakenet_ready"],
            logo_url=logo_url,
        )
        gr.HTML(nav_html)

        # Navigation buttons (styled as pills via JS)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    scan_nav = gr.Button("SCAN", size="sm", elem_classes=["nav-pill", "nav-pill-active"])
                    history_nav = gr.Button("HISTORY", size="sm", elem_classes=["nav-pill"])
                    settings_nav = gr.Button("SETTINGS", size="sm", elem_classes=["nav-pill"])

        # ===== MAIN CONTENT AREA =====

        # ──── PAGE: SCAN ────
        scan_page = gr.Column(visible=True, elem_classes=["page-transition"])
        with scan_page:
            with gr.Tabs():
                # ===== IMAGE TAB =====
                with gr.TabItem("Image"):
                    with gr.Row():
                        # Upload panel (left ~35%)
                        with gr.Column(scale=2, elem_classes=["panel-left"]):
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
                                "Analyze", variant="primary", size="lg",
                                elem_classes=["analyze-btn"],
                            )
                            heatmap_toggle = gr.Checkbox(
                                label="Show Artifact Heatmap", value=True,
                            )

                        # Results panel (right ~65%)
                        with gr.Column(scale=3, elem_classes=["panel-right"]):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    img_gauge = gr.HTML(
                                        value=generate_empty_state("Upload an image and click Analyze"),
                                        elem_classes=["gauge-container"],
                                    )
                                    img_agreement = gr.HTML()

                                with gr.Column(scale=1):
                                    img_display = gr.Image(
                                        label="Preview", type="pil",
                                        elem_classes=["center-display"],
                                    )

                            img_scores = gr.HTML(elem_classes=["scores-container"])
                            img_verdict = gr.HTML(elem_classes=["verdict-container"])
                            img_exif = gr.HTML()

                            # Radar chart
                            img_radar = gr.HTML(
                                value="",
                                elem_classes=["radar-container"],
                            )

                            with gr.Accordion("Raw Details", open=False):
                                img_details = gr.Textbox(lines=10, interactive=False, show_label=False)
                            download_report_btn = gr.Button(
                                "Download Report", variant="secondary", visible=False,
                            )
                            report_file = gr.File(label="Report", visible=False)

                    # States
                    img_gradcam_state = gr.State(value=None)
                    img_original_state = gr.State(value=None)
                    img_result_state = gr.State(value=None)

                    def _show_loading():
                        return (
                            generate_skeleton_gauge(),
                            "",
                            generate_skeleton_bars(),
                            generate_skeleton_verdict(),
                            "",
                            "",
                        )

                    def _run_image(image, mode):
                        if image is None:
                            empty = generate_empty_state("Upload an image to begin analysis")
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

                    # Show skeleton loading, then run analysis
                    analyze_btn.click(
                        fn=_show_loading,
                        outputs=[img_gauge, img_agreement, img_scores, img_verdict,
                                 img_exif, img_details],
                    ).then(
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
                        with gr.Column(scale=2, elem_classes=["panel-left"]):
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
                                "Analyze", variant="primary", size="lg",
                                elem_classes=["analyze-btn"],
                            )

                        with gr.Column(scale=3, elem_classes=["panel-right"]):
                            gradcam_video_output = gr.Video(
                                label="GradCAM Detection Output",
                                elem_classes=["center-display"],
                            )
                            vid_gauge = gr.HTML(
                                value=generate_empty_state("Upload a video and click Analyze"),
                                elem_classes=["gauge-container"],
                            )
                            vid_summary = gr.HTML(elem_classes=["scores-container"])
                            vid_verdict = gr.HTML(elem_classes=["verdict-container"])
                            vid_timeline = gr.HTML(
                                value="",
                                elem_classes=["timeline-container"],
                            )
                            with gr.Accordion("Frame Details", open=False):
                                vid_frames = gr.Textbox(lines=12, interactive=False, show_label=False)

                    def _run_video(video, fps, aggregation, progress=gr.Progress()):
                        if video is None:
                            empty = generate_empty_state("Upload a video to begin analysis")
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
                        <div style="padding:12px 0;color:#A1A1AA;font-size:0.82rem;line-height:1.7;">
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
                        with gr.Column(scale=2, elem_classes=["panel-left"]):
                            input_audio = gr.Audio(
                                type="filepath", label="Upload Audio",
                                elem_classes=["upload-area"],
                            )
                            audio_btn = gr.Button(
                                "Analyze", variant="primary", size="lg",
                                elem_classes=["analyze-btn"],
                            )

                        with gr.Column(scale=3, elem_classes=["panel-right"]):
                            aud_gauge = gr.HTML(
                                value=generate_empty_state("Upload an audio file and click Analyze"),
                                elem_classes=["gauge-container"],
                            )
                            audio_info = gr.HTML(
                                value='<div style="color:#52525B;font-size:0.78rem;padding:8px 0;">'
                                      'Supported: WAV, MP3, FLAC, M4A, OGG, AAC, WMA</div>',
                            )
                            aud_details = gr.HTML(elem_classes=["scores-container"])
                            aud_verdict = gr.HTML(elem_classes=["verdict-container"])

                    def _run_audio(audio, progress=gr.Progress()):
                        if audio is None:
                            empty = generate_empty_state("Upload audio to begin analysis")
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
                        <div style="padding:12px 0;color:#A1A1AA;font-size:0.82rem;line-height:1.7;">
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
                        with gr.Column(scale=2, elem_classes=["panel-left"]):
                            mm_image = gr.Image(type="pil", label="Image (optional)", elem_classes=["upload-area"])
                            mm_video = gr.Video(label="Video (optional)", elem_classes=["upload-area"])
                            mm_audio = gr.Audio(type="filepath", label="Audio (optional)", elem_classes=["upload-area"])
                            mm_btn = gr.Button(
                                "Analyze All", variant="primary", size="lg",
                                elem_classes=["analyze-btn"],
                            )

                        with gr.Column(scale=3, elem_classes=["panel-right"]):
                            mm_gauge = gr.HTML(
                                value=generate_empty_state("Upload one or more media types for cross-modal fusion"),
                                elem_classes=["gauge-container"],
                            )
                            mm_bars = gr.HTML(elem_classes=["scores-container"])
                            mm_verdict = gr.HTML(elem_classes=["verdict-container"])
                            with gr.Accordion("Raw JSON", open=False):
                                mm_json = gr.Textbox(lines=10, interactive=False, show_label=False)

                    def _run_multimodal(image, video, audio, progress=gr.Progress()):
                        if image is None and video is None and audio is None:
                            empty = generate_empty_state("Upload media to begin analysis")
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
        history_page = gr.Column(visible=False, elem_classes=["page-transition"])
        with history_page:
            gr.HTML('<h2 style="color:#FAFAFA;margin-bottom:16px;font-weight:700;">Analysis History</h2>')
            history_html = gr.HTML(
                value=generate_history_html(history_db.get_recent(20)),
            )
            refresh_btn = gr.Button("Refresh", variant="secondary")
            refresh_btn.click(
                fn=lambda: generate_history_html(history_db.get_recent(20)),
                outputs=[history_html],
            )

        # ──── PAGE: SETTINGS ────
        settings_page = gr.Column(visible=False, elem_classes=["page-transition"])
        with settings_page:
            gr.HTML('<h2 style="color:#FAFAFA;margin-bottom:16px;font-weight:700;">Settings</h2>')

            # Modules panel
            modules_html = generate_modules_panel(status["loaded"])
            gr.HTML(modules_html)

            gr.HTML(f"""
            <div style="padding:16px;background:#18181B;
                        border:1px solid rgba(255,255,255,0.06);border-radius:12px;margin-top:16px;">
                <div style="color:#A1A1AA;font-size:0.82rem;line-height:2.2;">
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.03);">
                        <span style="color:#71717A;font-weight:500;">Session ID</span>
                        <code style="color:#6366F1;font-size:0.78rem;">{session_id}</code>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.03);">
                        <span style="color:#71717A;font-weight:500;">Models Loaded</span>
                        <span>{len(status['loaded'])}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.03);">
                        <span style="color:#71717A;font-weight:500;">Models Missing</span>
                        <span>{len(status['missing'])}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.03);">
                        <span style="color:#71717A;font-weight:500;">CorefakeNet</span>
                        <span style="color:{'#22C55E' if status['corefakenet_ready'] else '#71717A'};">{'Ready' if status['corefakenet_ready'] else 'Not loaded'}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.03);">
                        <span style="color:#71717A;font-weight:500;">Device</span>
                        <span>CPU</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:4px 0;">
                        <span style="color:#71717A;font-weight:500;">API</span>
                        <span>/api/v1 (REST) | /docs (Swagger)</span>
                    </div>
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
            ProofyX v2.0 &mdash; AI-Powered Media Forensics
        </div>
        """)

    return demo
