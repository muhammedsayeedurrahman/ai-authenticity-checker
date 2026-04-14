"""
ProofyX UI HTML generator functions.

Pure functions that take data and return HTML strings.
No Gradio or framework dependencies — just HTML/CSS/SVG.
"""

from __future__ import annotations

import json
from typing import Any, Optional


# ──────────────────────────────────────────────
# Risk Gauge (SVG circular arc)
# ──────────────────────────────────────────────

def generate_gauge_html(risk_pct: float, label: str = "Risk Score") -> str:
    """SVG circular gauge with animated arc stroke."""
    risk_pct = max(0, min(100, risk_pct))
    radius = 80
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - risk_pct / 100)

    if risk_pct > 70:
        color, glow = "#EC4899", "rgba(236,72,153,0.4)"
    elif risk_pct > 40:
        color, glow = "#F59E0B", "rgba(245,158,11,0.4)"
    else:
        color, glow = "#10B981", "rgba(16,185,129,0.4)"

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


# ──────────────────────────────────────────────
# Score Bars (per-model horizontal bars)
# ──────────────────────────────────────────────

def generate_score_bars_html(scores_dict: dict[str, float]) -> str:
    """Animated horizontal score bars for per-model breakdown."""
    if not scores_dict:
        return '<div style="color:#64748B;text-align:center;padding:16px;">No scores available</div>'

    bars_html = ""
    for name, value in scores_dict.items():
        pct = max(0, min(100, value * 100))
        if pct > 70:
            color, glow = "#EC4899", "rgba(236,72,153,0.3)"
        elif pct > 40:
            color, glow = "#F59E0B", "rgba(245,158,11,0.3)"
        else:
            color, glow = "#10B981", "rgba(16,185,129,0.3)"

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


# ──────────────────────────────────────────────
# Verdict Card
# ──────────────────────────────────────────────

def generate_verdict_html(verdict_str: str) -> str:
    """Color-coded verdict badge card with 4-tier system."""
    if not verdict_str:
        return ""

    upper = verdict_str.upper()
    if "LIKELY MANIPULATED" in upper:
        bg, border, color = "rgba(236,72,153,0.1)", "rgba(236,72,153,0.3)", "#EC4899"
        icon = "&#9888;"
    elif "POSSIBLY MANIPULATED" in upper:
        bg, border, color = "rgba(245,158,11,0.1)", "rgba(245,158,11,0.3)", "#F59E0B"
        icon = "&#9888;"
    elif "UNCERTAIN" in upper:
        bg, border, color = "rgba(148,163,184,0.1)", "rgba(148,163,184,0.3)", "#94A3B8"
        icon = "&#63;"
    elif "LIKELY AUTHENTIC" in upper:
        bg, border, color = "rgba(16,185,129,0.1)", "rgba(16,185,129,0.3)", "#10B981"
        icon = "&#10003;"
    # Legacy fallback
    elif "HIGH" in upper or "CRITICAL" in upper:
        bg, border, color = "rgba(236,72,153,0.1)", "rgba(236,72,153,0.3)", "#EC4899"
        icon = "&#9888;"
    elif "MEDIUM" in upper:
        bg, border, color = "rgba(245,158,11,0.1)", "rgba(245,158,11,0.3)", "#F59E0B"
        icon = "&#9888;"
    else:
        bg, border, color = "rgba(16,185,129,0.1)", "rgba(16,185,129,0.3)", "#10B981"
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


# ──────────────────────────────────────────────
# System Header
# ──────────────────────────────────────────────

def generate_system_header(
    loaded_models: list[str],
    session_id: str = "",
    corefakenet_ready: bool = False,
) -> str:
    """System status header with model count and session ID."""
    count = len(loaded_models)
    fast_tag = ' <span style="color:#A855F7;">| Fast Mode</span>' if corefakenet_ready else ""
    sid = f'<span style="color:#64748B;font-size:0.72rem;margin-left:12px;">SID: {session_id[:8]}</span>' if session_id else ""

    return f"""
    <div class="system-header">
        <div>
            <h1>Ingestion Terminal</h1>
            <span style="color:#64748B;font-size:0.8rem;">{count} detection modules active{fast_tag}</span>
        </div>
        <div class="system-status">
            <span class="pulse-dot"></span>
            SYSTEM ACTIVE {sid}
        </div>
    </div>
    """


# ──────────────────────────────────────────────
# Detection Modules Panel
# ──────────────────────────────────────────────

def generate_modules_panel(
    loaded_models: list[str],
    all_model_names: Optional[list[str]] = None,
) -> str:
    """Show loaded model status with green/gray indicators."""
    if all_model_names is None:
        all_model_names = [
            "ViT Deepfake", "EfficientNet-B4 Texture", "Frequency CNN",
            "Face Deepfake", "DINOv2", "EfficientNet Auth",
            "Fusion MLP", "CorefakeNet", "Audio CNN",
        ]

    loaded_lower = [m.lower() for m in loaded_models]
    items = ""
    for name in all_model_names:
        is_active = any(
            keyword in name.lower()
            for loaded_name in loaded_lower
            for keyword in loaded_name.split()
            if len(keyword) > 3
        ) or name.lower().replace(" ", "") in "".join(loaded_lower).replace(" ", "")

        dot_class = "module-dot-active" if is_active else "module-dot-inactive"
        status_text = "Active" if is_active else "Inactive"
        text_color = "#E2E8F0" if is_active else "#64748B"

        items += f"""
        <div class="module-item">
            <span class="{dot_class}"></span>
            <span style="color:{text_color};">{name}</span>
            <span style="color:#64748B;font-size:0.72rem;margin-left:auto;">{status_text}</span>
        </div>"""

    return f"""
    <div style="padding:12px 0;">
        <div style="color:#94A3B8;font-size:0.78rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            Detection Modules
        </div>
        {items}
    </div>
    """


# ──────────────────────────────────────────────
# Model Agreement
# ──────────────────────────────────────────────

def generate_agreement_html(agreement: str) -> str:
    """Model agreement indicator (e.g., '5/7 models detect manipulation')."""
    if not agreement:
        return ""
    return f"""
    <div style="text-align:center;padding:8px 16px;margin:8px 0;
                background:rgba(255,255,255,0.03);border-radius:8px;
                border:1px solid rgba(255,255,255,0.06);
                color:#94A3B8;font-size:0.82rem;font-weight:500;">
        {agreement}
    </div>
    """


# ──────────────────────────────────────────────
# EXIF Metadata Card
# ──────────────────────────────────────────────

def generate_exif_html(metadata: dict[str, Any]) -> str:
    """Render EXIF metadata as a collapsible card."""
    if not metadata or not metadata.get("has_exif"):
        findings = metadata.get("exif_findings", []) if metadata else []
        findings_html = "".join(
            f'<div style="color:#F59E0B;font-size:0.78rem;padding:2px 0;">&#9888; {f}</div>'
            for f in findings
        )
        return f"""
        <div style="padding:12px;background:rgba(245,158,11,0.05);
                    border:1px solid rgba(245,158,11,0.15);border-radius:10px;">
            <div style="color:#F59E0B;font-weight:600;font-size:0.82rem;margin-bottom:4px;">
                No EXIF Metadata
            </div>
            {findings_html}
        </div>
        """

    exif = metadata.get("exif", {}) or {}
    rows = ""
    for key, val in exif.items():
        if val:
            rows += f"""
            <div style="display:flex;justify-content:space-between;padding:4px 0;
                        border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94A3B8;font-size:0.78rem;">{key}</span>
                <span style="color:#E2E8F0;font-size:0.78rem;">{val}</span>
            </div>"""

    findings = metadata.get("exif_findings", [])
    findings_html = "".join(
        f'<div style="color:#F59E0B;font-size:0.78rem;padding:2px 0;">&#9888; {f}</div>'
        for f in findings
    )

    return f"""
    <div style="padding:12px;">
        <div style="color:#94A3B8;font-size:0.78rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            EXIF Metadata
        </div>
        {rows}
        {findings_html}
    </div>
    """


# ──────────────────────────────────────────────
# History Table
# ──────────────────────────────────────────────

def generate_history_html(entries: list[dict[str, Any]]) -> str:
    """Render analysis history as an HTML table."""
    if not entries:
        return '<div style="color:#64748B;text-align:center;padding:32px;">No analysis history yet</div>'

    rows = ""
    for entry in entries:
        risk = entry.get("risk_score", 0) * 100
        verdict = entry.get("verdict", "")
        if risk > 70:
            badge_color = "#EC4899"
        elif risk > 40:
            badge_color = "#F59E0B"
        else:
            badge_color = "#10B981"

        rows += f"""
        <tr>
            <td><code style="color:#00F0FF;font-size:0.75rem;">{entry.get('id', '')}</code></td>
            <td>{entry.get('timestamp', '')[:19]}</td>
            <td>{entry.get('media_type', '').upper()}</td>
            <td><span style="color:{badge_color};font-weight:600;">{risk:.1f}%</span></td>
            <td><span style="color:{badge_color};">{verdict}</span></td>
            <td>{entry.get('file_name', '')}</td>
        </tr>"""

    return f"""
    <table class="history-table">
        <thead>
            <tr>
                <th>ID</th>
                <th>Timestamp</th>
                <th>Type</th>
                <th>Risk</th>
                <th>Verdict</th>
                <th>File</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


# ──────────────────────────────────────────────
# Utility: Parse model scores from raw details
# ──────────────────────────────────────────────

def parse_model_scores(details_str: str) -> dict[str, float]:
    """Extract {name: float} from raw details text."""
    scores = {}
    if not details_str:
        return scores

    skip_prefixes = (
        "---", "Face", "Fusion", "Analysis", "Active",
        "Learned", "Final", "Temperature", "Confidence",
    )

    for line in details_str.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        if any(line.startswith(p) for p in skip_prefixes):
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
