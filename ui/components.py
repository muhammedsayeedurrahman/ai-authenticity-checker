"""
ProofyX UI HTML generator functions.

Pure functions that take data and return HTML strings.
No Gradio or framework dependencies -- just HTML/CSS/SVG.
Uses the "Quiet Confidence" design system tokens.
"""

from __future__ import annotations

import json
from typing import Any, Optional


# ──────────────────────────────────────────────
# SVG Icons (Lucide-style, inline)
# ──────────────────────────────────────────────

_ICON_CHECK = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>'
)
_ICON_ALERT = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
)
_ICON_HELP = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
    '<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
)
_ICON_SHIELD = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
)


# ──────────────────────────────────────────────
# Risk Gauge (SVG circular arc)
# ──────────────────────────────────────────────

def generate_gauge_html(risk_pct: float, label: str = "Risk Score") -> str:
    """SVG circular gauge with animated arc stroke and ARIA support."""
    risk_pct = max(0, min(100, risk_pct))
    radius = 80
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - risk_pct / 100)

    if risk_pct > 70:
        color = "#EF4444"
    elif risk_pct > 40:
        color = "#EAB308"
    else:
        color = "#22C55E"

    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:24px 0;"
         role="meter" aria-label="{label}" aria-valuenow="{risk_pct:.0f}"
         aria-valuemin="0" aria-valuemax="100">
        <div class="card-3d" style="display:inline-block;">
            <div class="card-3d-inner" style="padding:8px;">
                <svg width="200" height="200" viewBox="0 0 200 200" aria-hidden="true">
                    <circle cx="100" cy="100" r="{radius}" fill="none"
                            stroke="rgba(255,255,255,0.04)" stroke-width="10"/>
                    <circle cx="100" cy="100" r="{radius}" fill="none"
                            stroke="{color}" stroke-width="10"
                            stroke-linecap="round"
                            stroke-dasharray="{circumference}"
                            stroke-dashoffset="{offset}"
                            transform="rotate(-90 100 100)"
                            style="--gauge-circumference:{circumference};--gauge-offset:{offset};
                                   animation:gauge-draw 1s ease-out;
                                   transition:stroke-dashoffset 0.6s ease;"/>
                    <text x="100" y="90" text-anchor="middle" fill="{color}"
                          font-size="38" font-weight="800" font-family="Inter,sans-serif">
                        {risk_pct:.0f}%
                    </text>
                    <text x="100" y="115" text-anchor="middle" fill="#71717A"
                          font-size="11" font-weight="500" font-family="Inter,sans-serif"
                          text-transform="uppercase" letter-spacing="0.1em">
                        {label}
                    </text>
                </svg>
            </div>
        </div>
    </div>
    """


# ──────────────────────────────────────────────
# Score Bars (per-model horizontal bars)
# ──────────────────────────────────────────────

def generate_score_bars_html(scores_dict: dict[str, float]) -> str:
    """Animated horizontal score bars for per-model breakdown."""
    if not scores_dict:
        return '<div style="color:#71717A;text-align:center;padding:16px;font-size:0.82rem;">No scores available</div>'

    bars_html = ""
    delay_ms = 0
    for name, value in scores_dict.items():
        pct = max(0, min(100, value * 100))
        if pct > 70:
            color = "#EF4444"
        elif pct > 40:
            color = "#EAB308"
        else:
            color = "#22C55E"

        bars_html += f"""
        <div style="margin-bottom:10px;animation:slide-up 0.4s ease-out {delay_ms}ms both;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#A1A1AA;font-size:0.78rem;font-weight:500;">{name}</span>
                <span style="color:{color};font-size:0.78rem;font-weight:700;">{pct:.1f}%</span>
            </div>
            <div style="height:6px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{color};border-radius:3px;
                            --fill-width:{pct}%;
                            animation:reveal-bar 0.7s ease-out {delay_ms}ms both;"></div>
            </div>
        </div>
        """
        delay_ms += 100

    return f'<div style="padding:12px 0;">{bars_html}</div>'


# ──────────────────────────────────────────────
# Verdict Card
# ──────────────────────────────────────────────

def generate_verdict_html(verdict_str: str) -> str:
    """Minimal verdict badge with SVG icon and 4-tier color system."""
    if not verdict_str:
        return ""

    upper = verdict_str.upper()
    if "LIKELY MANIPULATED" in upper:
        bg = "rgba(239,68,68,0.08)"
        border = "rgba(239,68,68,0.2)"
        color = "#EF4444"
        icon = _ICON_ALERT
        level = "LIKELY MANIPULATED"
    elif "POSSIBLY MANIPULATED" in upper:
        bg = "rgba(234,179,8,0.08)"
        border = "rgba(234,179,8,0.2)"
        color = "#EAB308"
        icon = _ICON_ALERT
        level = "POSSIBLY MANIPULATED"
    elif "UNCERTAIN" in upper:
        bg = "rgba(113,113,122,0.08)"
        border = "rgba(113,113,122,0.2)"
        color = "#A1A1AA"
        icon = _ICON_HELP
        level = "UNCERTAIN"
    elif "LIKELY AUTHENTIC" in upper:
        bg = "rgba(34,197,94,0.08)"
        border = "rgba(34,197,94,0.2)"
        color = "#22C55E"
        icon = _ICON_CHECK
        level = "LIKELY AUTHENTIC"
    elif "HIGH" in upper or "CRITICAL" in upper:
        bg = "rgba(239,68,68,0.08)"
        border = "rgba(239,68,68,0.2)"
        color = "#EF4444"
        icon = _ICON_ALERT
        level = "HIGH RISK"
    elif "MEDIUM" in upper:
        bg = "rgba(234,179,8,0.08)"
        border = "rgba(234,179,8,0.2)"
        color = "#EAB308"
        icon = _ICON_ALERT
        level = "MEDIUM RISK"
    else:
        bg = "rgba(34,197,94,0.08)"
        border = "rgba(34,197,94,0.2)"
        color = "#22C55E"
        icon = _ICON_CHECK
        level = "LOW RISK"

    return f"""
    <div style="padding:14px 16px;border-radius:12px;
                background:{bg};border:1px solid {border};
                animation:slide-up 0.4s ease-out;margin-top:8px;"
         role="status" aria-label="Verdict: {level}">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="color:{color};display:flex;">{icon}</span>
            <span style="font-weight:700;font-size:0.78rem;color:{color};
                         text-transform:uppercase;letter-spacing:0.06em;">Verdict</span>
        </div>
        <div style="color:#A1A1AA;font-size:0.82rem;line-height:1.6;">{verdict_str}</div>
    </div>
    """


# ──────────────────────────────────────────────
# Top Navigation Bar
# ──────────────────────────────────────────────

def generate_top_nav(
    loaded_models: list[str],
    session_id: str = "",
    corefakenet_ready: bool = False,
    logo_url: str = "",
) -> str:
    """Top navigation bar: logo + nav pills + status."""
    count = len(loaded_models)
    fast_tag = ' <span style="color:#6366F1;font-size:0.7rem;">FAST</span>' if corefakenet_ready else ""

    logo_html = ""
    if logo_url:
        logo_html = f'<img src="{logo_url}" alt="ProofyX logo" />'

    return f"""
    <div class="top-nav-bar">
        <div class="top-nav-brand">
            {logo_html}
            <span>ProofyX</span>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="color:#71717A;font-size:0.72rem;">{count} models{fast_tag}</span>
        </div>
        <div class="top-nav-status">
            <span class="status-dot"></span>
            Online
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
    """Show loaded model status with minimal indicators."""
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
        text_color = "#FAFAFA" if is_active else "#52525B"

        items += f"""
        <div class="module-item">
            <span class="{dot_class}"></span>
            <span style="color:{text_color};font-size:0.78rem;">{name}</span>
        </div>"""

    return f"""
    <div style="padding:12px 0;">
        <div style="color:#71717A;font-size:0.7rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">
            Detection Modules
        </div>
        {items}
    </div>
    """


# ──────────────────────────────────────────────
# Model Agreement
# ──────────────────────────────────────────────

def generate_agreement_html(agreement: str) -> str:
    """Model agreement indicator."""
    if not agreement:
        return ""
    return f"""
    <div style="text-align:center;padding:8px 14px;margin:8px 0;
                background:rgba(255,255,255,0.02);border-radius:8px;
                border:1px solid rgba(255,255,255,0.04);
                color:#A1A1AA;font-size:0.78rem;font-weight:500;"
         role="status">
        {agreement}
    </div>
    """


# ──────────────────────────────────────────────
# EXIF Metadata Card
# ──────────────────────────────────────────────

def generate_exif_html(metadata: dict[str, Any]) -> str:
    """Render EXIF metadata as a clean key-value card."""
    if not metadata or not metadata.get("has_exif"):
        findings = metadata.get("exif_findings", []) if metadata else []
        findings_html = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;color:#EAB308;font-size:0.75rem;padding:2px 0;">'
            f'<span style="display:flex;">{_ICON_ALERT}</span>{f}</div>'
            for f in findings
        )
        return f"""
        <div style="padding:12px;background:rgba(234,179,8,0.04);
                    border:1px solid rgba(234,179,8,0.1);border-radius:10px;">
            <div style="color:#EAB308;font-weight:600;font-size:0.78rem;margin-bottom:4px;">
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
            <div style="display:flex;justify-content:space-between;padding:6px 0;
                        border-bottom:1px solid rgba(255,255,255,0.03);">
                <span style="color:#71717A;font-size:0.75rem;font-weight:500;">{key}</span>
                <span style="color:#A1A1AA;font-size:0.75rem;">{val}</span>
            </div>"""

    findings = metadata.get("exif_findings", [])
    findings_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;color:#EAB308;font-size:0.75rem;padding:3px 0;">'
        f'<span style="display:flex;">{_ICON_ALERT}</span>{f}</div>'
        for f in findings
    )

    return f"""
    <div style="padding:12px;animation:slide-up 0.4s ease-out;">
        <div style="color:#71717A;font-size:0.7rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">
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
    """Render analysis history as a modern table."""
    if not entries:
        return (
            '<div style="color:#71717A;text-align:center;padding:48px 24px;font-size:0.82rem;">'
            'No analysis history yet. Run a scan to see results here.</div>'
        )

    rows = ""
    for entry in entries:
        risk = entry.get("risk_score", 0) * 100
        verdict = entry.get("verdict", "")
        if risk > 70:
            badge_color = "#EF4444"
        elif risk > 40:
            badge_color = "#EAB308"
        else:
            badge_color = "#22C55E"

        rows += f"""
        <tr>
            <td><code style="color:#6366F1;font-size:0.72rem;background:rgba(99,102,241,0.08);
                            padding:2px 6px;border-radius:4px;">{entry.get('id', '')}</code></td>
            <td>{entry.get('timestamp', '')[:19]}</td>
            <td><span style="color:#71717A;font-size:0.72rem;text-transform:uppercase;
                            letter-spacing:0.04em;">{entry.get('media_type', '').upper()}</span></td>
            <td><span style="color:{badge_color};font-weight:600;">{risk:.1f}%</span></td>
            <td><span style="color:{badge_color};font-size:0.78rem;">{verdict}</span></td>
            <td style="color:#71717A;">{entry.get('file_name', '')}</td>
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
# Skeleton Loading States
# ──────────────────────────────────────────────

def generate_skeleton_gauge() -> str:
    """Skeleton placeholder for the risk gauge during loading."""
    return """
    <div style="display:flex;flex-direction:column;align-items:center;padding:24px 0;">
        <div class="skeleton" style="width:200px;height:200px;border-radius:50%;"></div>
    </div>
    """


def generate_skeleton_bars(count: int = 5) -> str:
    """Skeleton placeholder for score bars during loading."""
    bars = ""
    for i in range(count):
        width = 60 + (i * 7) % 30
        bars += f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <div class="skeleton" style="width:100px;height:12px;"></div>
                <div class="skeleton" style="width:40px;height:12px;"></div>
            </div>
            <div class="skeleton" style="height:6px;width:{width}%;border-radius:3px;"></div>
        </div>
        """
    return f'<div style="padding:12px 0;">{bars}</div>'


def generate_skeleton_verdict() -> str:
    """Skeleton placeholder for verdict card during loading."""
    return """
    <div style="padding:14px 16px;border-radius:12px;margin-top:8px;">
        <div class="skeleton" style="width:80px;height:14px;margin-bottom:8px;"></div>
        <div class="skeleton" style="width:100%;height:12px;margin-bottom:4px;"></div>
        <div class="skeleton" style="width:70%;height:12px;"></div>
    </div>
    """


# ──────────────────────────────────────────────
# Empty State
# ──────────────────────────────────────────────

def generate_empty_state(message: str = "Upload media to begin analysis") -> str:
    """Subtle guidance text for empty result panels."""
    return f"""
    <div style="text-align:center;padding:32px 16px;color:#52525B;font-size:0.82rem;">
        <div style="display:flex;justify-content:center;margin-bottom:12px;color:#3f3f46;">
            {_ICON_SHIELD}
        </div>
        {message}
    </div>
    """


# ──────────────────────────────────────────────
# Utility: Parse model scores from raw details
# ──────────────────────────────────────────────

def parse_model_scores(details_str: str) -> dict[str, float]:
    """Extract {{name: float}} from raw details text."""
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
