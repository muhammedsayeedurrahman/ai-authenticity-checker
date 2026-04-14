"""
PDF/HTML forensic report generation for ProofyX.

Uses Jinja2 for templating and WeasyPrint for PDF conversion.
All images embedded as base64 for self-contained reports.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from core.types import ForensicReport, ModelScore, ExifMetadata

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Verdict color mapping
VERDICT_COLORS = {
    "LIKELY MANIPULATED": {"bg": "#2d0a0a", "border": "#ef4444", "text": "#ef4444"},
    "POSSIBLY MANIPULATED": {"bg": "#2d1f0a", "border": "#f59e0b", "text": "#f59e0b"},
    "UNCERTAIN": {"bg": "#1a1a2e", "border": "#94a3b8", "text": "#94a3b8"},
    "LIKELY AUTHENTIC": {"bg": "#0a2d1a", "border": "#10b981", "text": "#10b981"},
}


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_forensic_report(
    analysis_result: dict[str, Any],
    file_name: str = "",
    image_pil: Optional[Image.Image] = None,
    gradcam_pil: Optional[Image.Image] = None,
    exif: Optional[ExifMetadata] = None,
) -> ForensicReport:
    """Build a ForensicReport dataclass from pipeline results."""
    report = ForensicReport(
        media_type=analysis_result.get("media_type", "image"),
        file_name=file_name,
        overall_verdict=analysis_result.get("verdict", ""),
        overall_confidence=_confidence_to_float(analysis_result.get("confidence", "")),
        risk_level=analysis_result.get("risk_level", ""),
        risk_score=analysis_result.get("risk_score", 0.0),
        fusion_mode=analysis_result.get("fusion_mode", ""),
        models_used=analysis_result.get("models_used", 0),
        face_detected=analysis_result.get("face_detected", False),
        processing_time_ms=analysis_result.get("processing_time_ms", 0.0),
        explanation=analysis_result.get("explanation", ""),
    )

    # Per-model scores
    raw_scores = analysis_result.get("model_scores", {})
    for name, prob in raw_scores.items():
        score_verdict = "SUSPICIOUS" if prob > 0.5 else "NORMAL"
        report.model_results[name] = ModelScore(
            name=name,
            probability=prob,
            verdict=score_verdict,
            confidence=abs(prob - 0.5) * 200,
            description=f"{name} detection score",
        )

    # Visuals
    if image_pil is not None:
        report.source_image_base64 = image_to_base64(image_pil, "JPEG")
        report.image_dimensions = (image_pil.width, image_pil.height)

    if gradcam_pil is not None:
        report.gradcam_overlay_base64 = image_to_base64(gradcam_pil)

    # EXIF
    if exif is not None:
        report.exif_metadata = exif

    # Video-specific
    if analysis_result.get("media_type") == "video":
        report.total_frames_analyzed = analysis_result.get("total_frames_analyzed", 0)
        report.fake_frames = analysis_result.get("fake_frames", 0)

    # Recommendations
    report.recommendations = _generate_recommendations(analysis_result)
    report.warnings = _generate_warnings(analysis_result)

    return report


def generate_html_report(report: ForensicReport) -> str:
    """Render forensic report as self-contained HTML."""
    try:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
        )
        template = env.get_template("forensic_report.html")
        return template.render(report=report, verdict_colors=VERDICT_COLORS)
    except ImportError:
        logger.warning("Jinja2 not installed, using inline template")
        return _render_inline_report(report)
    except Exception as e:
        logger.warning("Template rendering failed: %s, using inline", e)
        return _render_inline_report(report)


def generate_pdf_report(report: ForensicReport, output_path: str) -> str:
    """Generate PDF from forensic report. Returns output path."""
    html = generate_html_report(report)
    try:
        from weasyprint import HTML  # type: ignore
        HTML(string=html).write_pdf(output_path)
        return output_path
    except ImportError:
        logger.warning("WeasyPrint not installed, saving HTML instead")
        html_path = output_path.replace(".pdf", ".html")
        Path(html_path).write_text(html, encoding="utf-8")
        return html_path


def _confidence_to_float(conf: str) -> float:
    """Convert confidence string to 0-100 float."""
    mapping = {"HIGH": 90.0, "MEDIUM": 60.0, "LOW": 30.0}
    return mapping.get(conf.upper(), 50.0)


def _generate_recommendations(result: dict) -> list[str]:
    """Generate actionable recommendations based on analysis."""
    recs = []
    risk = result.get("risk_score", 0.0)
    verdict = result.get("verdict", "")

    if risk > 0.7:
        recs.append("This media shows strong indicators of AI generation or manipulation.")
        recs.append("Do not share as authentic content without further verification.")
        recs.append("Consider reverse image search to check for original sources.")
    elif risk > 0.45:
        recs.append("Some manipulation indicators detected. Exercise caution.")
        recs.append("Cross-reference with original source if available.")
    else:
        recs.append("No strong manipulation indicators detected.")
        recs.append("Analysis does not guarantee authenticity — exercise normal judgment.")

    if not result.get("face_detected", False) and result.get("media_type") == "image":
        recs.append("No face detected — face-specific analysis was skipped.")

    return recs


def _generate_warnings(result: dict) -> list[str]:
    """Generate warning messages based on analysis results."""
    warnings = []
    models_used = result.get("models_used", 0)

    if models_used < 3:
        warnings.append(f"Only {models_used} model(s) available — results may be less reliable.")

    if result.get("fusion_mode") == "weighted_avg":
        warnings.append("Learned fusion model not loaded — using fallback weighted averaging.")

    return warnings


def _render_inline_report(report: ForensicReport) -> str:
    """Fallback inline HTML rendering when Jinja2 is not available."""
    vc = VERDICT_COLORS.get(report.overall_verdict, VERDICT_COLORS["UNCERTAIN"])

    # Model scores rows
    model_rows = ""
    for name, ms in report.model_results.items():
        pct = ms.probability * 100
        bar_color = "#ef4444" if pct > 70 else "#f59e0b" if pct > 40 else "#10b981"
        model_rows += f"""
        <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #1e293b;color:#e2e8f0;">{name}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #1e293b;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="flex:1;height:6px;background:#1e293b;border-radius:3px;">
                        <div style="width:{pct:.0f}%;height:100%;background:{bar_color};border-radius:3px;"></div>
                    </div>
                    <span style="color:{bar_color};font-weight:600;min-width:50px;">{pct:.1f}%</span>
                </div>
            </td>
        </tr>"""

    # Images
    source_img = ""
    if report.source_image_base64:
        source_img = f'<img src="{report.source_image_base64}" style="max-width:100%;border-radius:8px;" />'

    gradcam_img = ""
    if report.gradcam_overlay_base64:
        gradcam_img = f'<img src="{report.gradcam_overlay_base64}" style="max-width:100%;border-radius:8px;" />'

    # EXIF section
    exif_section = ""
    if report.exif_metadata:
        exif = report.exif_metadata
        exif_rows = ""
        if exif.camera_make:
            exif_rows += f"<tr><td style='padding:6px;color:#94a3b8;'>Camera</td><td style='padding:6px;color:#e2e8f0;'>{exif.camera_make} {exif.camera_model or ''}</td></tr>"
        if exif.timestamp:
            exif_rows += f"<tr><td style='padding:6px;color:#94a3b8;'>Timestamp</td><td style='padding:6px;color:#e2e8f0;'>{exif.timestamp}</td></tr>"
        if exif.software:
            exif_rows += f"<tr><td style='padding:6px;color:#94a3b8;'>Software</td><td style='padding:6px;color:#e2e8f0;'>{exif.software}</td></tr>"
        if exif.gps_coordinates:
            exif_rows += f"<tr><td style='padding:6px;color:#94a3b8;'>GPS</td><td style='padding:6px;color:#e2e8f0;'>{exif.gps_coordinates}</td></tr>"
        for finding in exif.findings:
            exif_rows += f"<tr><td colspan='2' style='padding:6px;color:#f59e0b;'>&#9888; {finding}</td></tr>"
        if exif_rows:
            exif_section = f"""
            <div style="margin-top:24px;">
                <h2 style="color:#00f0ff;font-size:1.1rem;">EXIF Metadata</h2>
                <table style="width:100%;border-collapse:collapse;">{exif_rows}</table>
            </div>"""

    # Recommendations
    recs_html = "".join(f"<li style='color:#e2e8f0;margin:4px 0;'>{r}</li>" for r in report.recommendations)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ProofyX Forensic Report — {report.report_id}</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; padding: 32px; }}
    .container {{ max-width: 800px; margin: 0 auto; }}
    .header {{ text-align: center; margin-bottom: 32px; }}
    .header h1 {{ font-size: 1.8rem; background: linear-gradient(135deg, #00f0ff, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .verdict-badge {{ display: inline-block; padding: 10px 24px; border-radius: 50px; font-weight: 700; font-size: 1rem; margin: 16px 0; }}
    .section {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .stat {{ background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px; text-align: center; }}
    .stat-value {{ font-size: 1.4rem; font-weight: 800; color: #00f0ff; }}
    .stat-label {{ font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>PROOFYX FORENSIC REPORT</h1>
        <p style="color:#94a3b8;font-size:0.85rem;">Report ID: {report.report_id} | {report.timestamp}</p>
        <div class="verdict-badge" style="background:{vc['bg']};border:2px solid {vc['border']};color:{vc['text']};">
            {report.overall_verdict}
        </div>
    </div>

    <div class="section">
        <h2 style="color:#00f0ff;font-size:1.1rem;margin-bottom:12px;">Executive Summary</h2>
        <div class="grid">
            <div class="stat"><div class="stat-value">{report.risk_score * 100:.1f}%</div><div class="stat-label">Risk Score</div></div>
            <div class="stat"><div class="stat-value">{report.media_type.upper()}</div><div class="stat-label">Media Type</div></div>
            <div class="stat"><div class="stat-value">{report.models_used}</div><div class="stat-label">Models Used</div></div>
            <div class="stat"><div class="stat-value">{report.processing_time_ms:.0f}ms</div><div class="stat-label">Processing Time</div></div>
        </div>
    </div>

    <div class="section">
        <h2 style="color:#00f0ff;font-size:1.1rem;margin-bottom:12px;">Visual Evidence</h2>
        <div class="grid">
            <div>{source_img or '<p style="color:#64748b;">No source image</p>'}</div>
            <div>{gradcam_img or '<p style="color:#64748b;">No heatmap available</p>'}</div>
        </div>
    </div>

    <div class="section">
        <h2 style="color:#00f0ff;font-size:1.1rem;margin-bottom:12px;">Model Analysis</h2>
        <table style="width:100%;border-collapse:collapse;">{model_rows or '<tr><td style="color:#64748b;">No model data</td></tr>'}</table>
    </div>

    {exif_section}

    <div class="section">
        <h2 style="color:#00f0ff;font-size:1.1rem;margin-bottom:12px;">Recommendations</h2>
        <ul style="padding-left:20px;">{recs_html}</ul>
    </div>

    <div style="text-align:center;margin-top:32px;padding:16px;border-top:1px solid rgba(255,255,255,0.08);color:#64748b;font-size:0.75rem;">
        Generated by ProofyX v2.0 | {report.fusion_mode} fusion | {report.models_used} models
        <br>This report is for informational purposes only.
    </div>
</div>
</body>
</html>"""
