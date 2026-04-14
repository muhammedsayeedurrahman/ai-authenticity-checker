"""
ProofyX UI Theme — CSS, JS, and Gradio theme configuration.

Cybersecurity dashboard aesthetic: dark background, glassmorphism cards,
cyan/violet gradient accents, animated starfield, grid pattern.
"""

from __future__ import annotations

import gradio as gr

# ──────────────────────────────────────────────
# Force Dark Mode JS
# ──────────────────────────────────────────────

FORCE_DARK_JS = """
() => {
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
    document.body.style.backgroundColor = '#0A0E1A';
}
"""

# ──────────────────────────────────────────────
# Stars Background JS (injected via js_on_load)
# ──────────────────────────────────────────────

STARS_JS = """
(props) => {
    const container = props.element;
    if (!container) return;
    for (let i = 0; i < 120; i++) {
        const star = document.createElement('div');
        star.style.cssText = `
            position:fixed; width:${Math.random()*2+1}px; height:${Math.random()*2+1}px;
            background:white; border-radius:50%; opacity:${Math.random()*0.4+0.1};
            top:${Math.random()*100}%; left:${Math.random()*100}%;
            animation: star-twinkle ${Math.random()*4+2}s ease-in-out infinite;
            pointer-events:none; z-index:0;
        `;
        container.appendChild(star);
    }
}
"""

# ──────────────────────────────────────────────
# Chart.js CDN (injected via head param)
# ──────────────────────────────────────────────

CHARTJS_HEAD = '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>'

# ──────────────────────────────────────────────
# Radar Chart JS template
# ──────────────────────────────────────────────

RADAR_JS = """
(props) => {
    const el = props.element;
    if (!el || !props.value) return;
    const data = props.value;
    const canvas = el.querySelector('canvas');
    if (!canvas) return;

    if (canvas._chart) canvas._chart.destroy();

    canvas._chart = new Chart(canvas.getContext('2d'), {
        type: 'radar',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: 'Model Scores',
                data: Object.values(data).map(v => v * 100),
                borderColor: '#00F0FF',
                backgroundColor: 'rgba(0,240,255,0.1)',
                borderWidth: 2,
                pointBackgroundColor: '#00F0FF',
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { display: false },
                    pointLabels: { color: '#94A3B8', font: { size: 10 } },
                }
            },
            plugins: { legend: { display: false } },
        }
    });
}
"""

# ──────────────────────────────────────────────
# Timeline Chart JS template
# ──────────────────────────────────────────────

TIMELINE_JS = """
(props) => {
    const el = props.element;
    if (!el || !props.value) return;
    const data = props.value;
    const canvas = el.querySelector('canvas');
    if (!canvas) return;

    if (canvas._chart) canvas._chart.destroy();

    canvas._chart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: data.timestamps.map(t => t.toFixed(1) + 's'),
            datasets: [{
                label: 'Risk Score',
                data: data.scores,
                borderColor: '#00F0FF',
                backgroundColor: 'rgba(236,72,153,0.1)',
                fill: true,
                tension: 0.3,
                borderWidth: 2,
                pointRadius: 2,
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748B', font: { size: 9 } } },
                y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748B' } },
            },
            plugins: { legend: { display: false } },
        }
    });
}
"""

# ──────────────────────────────────────────────
# CSS — Full cybersecurity dashboard theme
# ──────────────────────────────────────────────

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
    max-width: 1600px !important;
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

/* ===== Stars Animation ===== */
@keyframes star-twinkle {
    0%, 100% { opacity: 0.1; }
    50% { opacity: 0.6; }
}

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

/* ===== Sidebar Navigation ===== */
.nav-sidebar {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
.nav-btn {
    width: 100% !important;
    text-align: left !important;
    padding: 12px 20px !important;
    border-radius: 10px !important;
    border: none !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    margin: 2px 0 !important;
}
.nav-btn:hover {
    background: rgba(0,240,255,0.06) !important;
    color: var(--accent-cyan) !important;
}
.nav-btn-active {
    background: linear-gradient(135deg, rgba(0,240,255,0.1), rgba(168,85,247,0.1)) !important;
    color: var(--accent-cyan) !important;
    border-left: 3px solid var(--accent-cyan) !important;
}

/* ===== System Header ===== */
.system-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    margin-bottom: 16px;
}
.system-header h1 {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.system-status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--accent-green);
    font-size: 0.82rem;
    font-weight: 600;
}
.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse-glow 2s infinite;
}

/* ===== Upload Zone ===== */
.upload-zone {
    border: 2px dashed rgba(0,240,255,0.3) !important;
    border-radius: 14px !important;
    background: rgba(0,240,255,0.02) !important;
    transition: all 0.3s ease !important;
    padding: 24px;
    text-align: center;
}
.upload-zone:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 20px rgba(0,240,255,0.15) !important;
}

/* ===== Detection Modules Panel ===== */
.module-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.82rem;
}
.module-dot-active {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 6px var(--accent-green);
}
.module-dot-inactive {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #475569;
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
    letter-spacing: 0.5px;
    text-transform: uppercase;
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

/* ===== Mode Toggle (Radio) ===== */
.mode-toggle .wrap {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
}

/* ===== 3-Column Layout ===== */
.panel-left { min-width: 280px !important; }
.panel-center { min-width: 300px !important; }
.panel-right { min-width: 280px !important; }

/* ===== Result HTML Containers ===== */
.gauge-container, .scores-container, .verdict-container,
.center-display, .radar-container, .timeline-container {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* ===== Scan Progress Animation ===== */
.scan-progress {
    position: relative;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
    margin: 12px 0;
}
.scan-line {
    position: absolute;
    top: 0; left: 0;
    width: 30%;
    height: 100%;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    animation: scan-sweep 2s ease-in-out infinite;
}
@keyframes scan-sweep {
    0% { left: -30%; }
    100% { left: 100%; }
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

/* ===== History Table ===== */
.history-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.history-table th {
    padding: 10px 12px;
    text-align: left;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border-subtle);
    font-weight: 600;
}
.history-table td {
    padding: 10px 12px;
    color: var(--text-primary);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.history-table tr:hover td {
    background: rgba(0,240,255,0.03);
}

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

# ──────────────────────────────────────────────
# Gradio Theme Object
# ──────────────────────────────────────────────

def create_theme() -> gr.themes.Base:
    """Create the ProofyX Gradio theme."""
    return gr.themes.Base(
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
