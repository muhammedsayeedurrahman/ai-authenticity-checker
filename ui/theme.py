"""
ProofyX UI Theme -- Minimal + Elegant + Deeptech design system.

Deep Indigo palette, Three.js neural mesh background, CSS 3D transforms,
refined micro-animations, proper responsive breakpoints.
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
    document.body.style.backgroundColor = '#09090B';

    /* Initialize 3D background after CDN scripts load */
    setTimeout(() => {
        if (document.getElementById('proofyx-neural-canvas')) return;
        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reducedMotion) return;

        if (typeof THREE !== 'undefined') {
            initThreeBg();
        } else if (typeof VANTA !== 'undefined') {
            initVantaBg();
        }
    }, 600);

    function initThreeBg() {
        const canvas = document.createElement('canvas');
        canvas.id = 'proofyx-neural-canvas';
        Object.assign(canvas.style, {
            position: 'fixed', top: '0', left: '0',
            width: '100vw', height: '100vh',
            zIndex: '0', pointerEvents: 'none', opacity: '0.18',
        });
        document.body.prepend(canvas);

        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        renderer.setSize(window.innerWidth, window.innerHeight);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
        camera.position.z = 300;

        const COUNT = 90;
        const positions = new Float32Array(COUNT * 3);
        const velocities = [];
        for (let i = 0; i < COUNT; i++) {
            positions[i*3]   = (Math.random()-0.5)*500;
            positions[i*3+1] = (Math.random()-0.5)*500;
            positions[i*3+2] = (Math.random()-0.5)*200;
            velocities.push({ x:(Math.random()-0.5)*0.3, y:(Math.random()-0.5)*0.3, z:(Math.random()-0.5)*0.1 });
        }

        const pointsGeo = new THREE.BufferGeometry();
        pointsGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const pointsMat = new THREE.PointsMaterial({
            color: 0x6366F1, size: 2.5, transparent: true, opacity: 0.7,
            blending: THREE.AdditiveBlending, depthWrite: false,
        });
        scene.add(new THREE.Points(pointsGeo, pointsMat));

        const lineMat = new THREE.LineBasicMaterial({
            color: 0x3f3f46, transparent: true, opacity: 0.15,
            blending: THREE.AdditiveBlending, depthWrite: false,
        });

        let mouse = { x: 0, y: 0 };
        document.addEventListener('mousemove', (e) => {
            mouse.x = (e.clientX/window.innerWidth-0.5)*2;
            mouse.y = -(e.clientY/window.innerHeight-0.5)*2;
        });

        let frameCount = 0;
        const THRESHOLD = 80;

        function animate() {
            requestAnimationFrame(animate);
            frameCount++;
            if (frameCount % 2 !== 0) return;

            const pos = pointsGeo.attributes.position.array;
            for (let i = 0; i < COUNT; i++) {
                pos[i*3]   += velocities[i].x + mouse.x*0.05;
                pos[i*3+1] += velocities[i].y + mouse.y*0.05;
                pos[i*3+2] += velocities[i].z;
                if (pos[i*3]>250) pos[i*3]=-250;
                if (pos[i*3]<-250) pos[i*3]=250;
                if (pos[i*3+1]>250) pos[i*3+1]=-250;
                if (pos[i*3+1]<-250) pos[i*3+1]=250;
            }
            pointsGeo.attributes.position.needsUpdate = true;

            if (frameCount % 6 === 0) {
                scene.children.forEach(c => { if (c.isLineSegments) scene.remove(c); });
                const lp = [];
                for (let i = 0; i < COUNT; i++) {
                    for (let j = i+1; j < COUNT; j++) {
                        const dx=pos[i*3]-pos[j*3], dy=pos[i*3+1]-pos[j*3+1], dz=pos[i*3+2]-pos[j*3+2];
                        if (Math.sqrt(dx*dx+dy*dy+dz*dz) < THRESHOLD) {
                            lp.push(pos[i*3],pos[i*3+1],pos[i*3+2], pos[j*3],pos[j*3+1],pos[j*3+2]);
                        }
                    }
                }
                if (lp.length > 0) {
                    const lg = new THREE.BufferGeometry();
                    lg.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
                    scene.add(new THREE.LineSegments(lg, lineMat));
                }
            }
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth/window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    function initVantaBg() {
        const el = document.createElement('div');
        el.id = 'proofyx-neural-canvas';
        Object.assign(el.style, {
            position: 'fixed', top: '0', left: '0',
            width: '100vw', height: '100vh',
            zIndex: '0', pointerEvents: 'none', opacity: '0.18',
        });
        document.body.prepend(el);
        VANTA.NET({
            el, THREE: typeof THREE !== 'undefined' ? THREE : undefined,
            mouseControls: true, touchControls: false,
            minHeight: 200, minWidth: 200, scale: 1.0, scaleMobile: 0.5,
            color: 0x6366F1, backgroundColor: 0x09090B,
            points: 8, maxDistance: 20, spacing: 16,
        });
    }
}
"""

# ──────────────────────────────────────────────
# CDN Head: Three.js + Vanta.js + Chart.js
# ──────────────────────────────────────────────

CDN_HEAD = """
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>
"""

# ──────────────────────────────────────────────
# Radar Chart JS (updated colors)
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
                borderColor: '#6366F1',
                backgroundColor: 'rgba(99,102,241,0.08)',
                borderWidth: 2,
                pointBackgroundColor: '#6366F1',
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
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { display: false },
                    pointLabels: { color: '#A1A1AA', font: { size: 10, family: 'Inter' } },
                }
            },
            plugins: { legend: { display: false } },
        }
    });
}
"""

# ──────────────────────────────────────────────
# Timeline Chart JS (updated colors)
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
                borderColor: '#6366F1',
                backgroundColor: 'rgba(239,68,68,0.06)',
                fill: true,
                tension: 0.3,
                borderWidth: 2,
                pointRadius: 2,
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#71717A', font: { size: 9 } } },
                y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#71717A' } },
            },
            plugins: { legend: { display: false } },
        }
    });
}
"""

# ──────────────────────────────────────────────
# CSS -- "Quiet Confidence" Design System
# ──────────────────────────────────────────────

CUSTOM_CSS = """
/* ===== Design Tokens ===== */
:root {
    --bg-void: #09090B;
    --bg-surface: #18181B;
    --bg-elevated: #27272A;
    --accent: #6366F1;
    --accent-glow: rgba(99,102,241,0.15);
    --success: #22C55E;
    --warning: #EAB308;
    --danger: #EF4444;
    --text-primary: #FAFAFA;
    --text-secondary: #A1A1AA;
    --text-muted: #71717A;
    --border: rgba(255,255,255,0.06);
    --border-hover: rgba(99,102,241,0.25);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

/* ===== Global Reset ===== */
body, .gradio-container, .dark {
    background-color: var(--bg-void);
    color: var(--text-primary);
}
.gradio-container {
    max-width: 1400px;
    margin: 0 auto;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    position: relative;
    z-index: 1;
}

footer { display: none; }

/* ===== Typography ===== */
h1, h2, h3 {
    color: var(--text-primary);
    font-weight: 700;
    line-height: 1.3;
}
h1 { font-size: 1.5rem; }
h2 { font-size: 1.25rem; }
h3 { font-size: 1rem; }

/* ===== Cards / Blocks ===== */
.block, .form, .panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.block:hover, .form:hover {
    border-color: var(--border-hover);
}

/* 3D Card Tilt */
.card-3d {
    perspective: 1000px;
}
.card-3d-inner {
    transition: transform 0.3s ease-out, box-shadow 0.3s ease-out;
}
.card-3d-inner:hover {
    transform: rotateX(2deg) rotateY(-2deg) translateZ(10px);
    box-shadow: -4px 4px 24px rgba(99,102,241,0.08);
}

/* ===== Top Navigation Bar ===== */
.top-nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 20px;
    gap: 16px;
}
.top-nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.top-nav-brand img {
    width: 32px;
    height: 32px;
    border-radius: 8px;
}
.top-nav-brand span {
    font-weight: 800;
    font-size: 1.1rem;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.top-nav-pills {
    display: flex;
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 3px;
}
.nav-pill {
    padding: 8px 20px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.2s ease;
}
.nav-pill:hover {
    color: var(--text-primary);
    background: rgba(255,255,255,0.04);
}
.nav-pill-active {
    background: var(--accent);
    color: #fff;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}
.top-nav-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: var(--success);
    font-weight: 600;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse-soft 2.5s ease-in-out infinite;
}

/* ===== Upload Zone ===== */
.upload-area .image-container,
.upload-area .video-container,
.upload-area .audio-container,
.upload-area [data-testid="image"],
.upload-area [data-testid="droparea"] {
    border: 1.5px dashed rgba(99,102,241,0.25);
    border-radius: var(--radius-md);
    background: rgba(99,102,241,0.02);
    transition: all 0.25s ease;
}
.upload-area:hover [data-testid="image"],
.upload-area:hover [data-testid="droparea"],
.upload-area:hover .image-container {
    border-color: var(--accent);
    box-shadow: 0 0 16px var(--accent-glow);
}

/* ===== Buttons ===== */
.analyze-btn, button.primary {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: var(--radius-md);
    padding: 12px 28px;
    font-weight: 700;
    font-size: 0.875rem;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 12px rgba(99,102,241,0.25);
    transition: all 0.2s ease;
    cursor: pointer;
}
.analyze-btn:hover, button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    background: #5457e5;
}

/* Secondary buttons */
button.secondary {
    background: var(--bg-elevated);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 10px 20px;
    font-weight: 600;
    font-size: 0.8rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
button.secondary:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

/* ===== Input Elements ===== */
input, textarea, select, .wrap {
    background: rgba(255,255,255,0.03);
    border-color: var(--border);
    color: var(--text-primary);
    border-radius: var(--radius-sm);
}
input:focus, textarea:focus, select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
}
label, .label-wrap span {
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.8rem;
}

/* ===== Tab Pills ===== */
.tab-nav {
    background: rgba(255,255,255,0.02);
    border-radius: var(--radius-md);
    padding: 4px;
    border: 1px solid var(--border);
    gap: 4px;
}
.tab-nav button {
    font-weight: 600;
    font-size: 0.8rem;
    color: var(--text-secondary);
    border-radius: var(--radius-sm);
    padding: 10px 24px;
    border: none;
    background: transparent;
    transition: all 0.2s ease;
}
.tab-nav button:hover {
    color: var(--text-primary);
    background: rgba(255,255,255,0.04);
}
.tab-nav button.selected {
    background: var(--accent);
    color: #fff;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
    border: none;
}

/* ===== Mode Toggle ===== */
.mode-toggle .wrap {
    background: rgba(255,255,255,0.02);
    border-radius: var(--radius-sm);
}

/* ===== Accordion ===== */
.accordion {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
}

/* ===== Gauge / Scores / Verdict containers ===== */
.gauge-container, .scores-container, .verdict-container,
.center-display, .radar-container, .timeline-container {
    padding: 0;
    border: none;
    background: transparent;
}

/* ===== Skeleton Loading ===== */
@keyframes skeleton-shimmer {
    0% { background-position: -200px 0; }
    100% { background-position: calc(200px + 100%) 0; }
}
.skeleton {
    background: linear-gradient(90deg, var(--bg-surface) 0%, var(--bg-elevated) 50%, var(--bg-surface) 100%);
    background-size: 200px 100%;
    animation: skeleton-shimmer 1.5s ease-in-out infinite;
    border-radius: var(--radius-sm);
}

/* ===== Progress Override ===== */
.progress-bar {
    background: var(--accent);
}
.eta-bar {
    background: var(--accent-glow);
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

/* ===== History Table ===== */
.history-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.history-table th {
    padding: 12px 14px;
    text-align: left;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.history-table td {
    padding: 12px 14px;
    color: var(--text-secondary);
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.history-table tr:hover td {
    background: rgba(99,102,241,0.03);
    color: var(--text-primary);
}

/* ===== Footer ===== */
.proofyx-footer {
    text-align: center;
    padding: 24px 0 12px 0;
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 500;
    border-top: 1px solid var(--border);
    margin-top: 32px;
    letter-spacing: 0.02em;
}

/* ===== Module Items ===== */
.module-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    font-size: 0.8rem;
}
.module-dot-active {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--success);
}
.module-dot-inactive {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #3f3f46;
}

/* ===== Entry Animations ===== */
@keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slide-up {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes scale-in {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}

/* Results animations */
@keyframes reveal-bar {
    from { width: 0; }
    to { width: var(--fill-width); }
}
@keyframes gauge-draw {
    from { stroke-dashoffset: var(--gauge-circumference); }
    to { stroke-dashoffset: var(--gauge-offset); }
}
@keyframes pulse-soft {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Page transitions */
@keyframes page-enter {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.page-transition {
    animation: page-enter 0.3s ease-out;
}

/* ===== Reduced Motion ===== */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    #proofyx-neural-canvas {
        display: none !important;
    }
}

/* ===== Responsive: Mobile ===== */
@media (max-width: 640px) {
    .gradio-container {
        padding: 8px;
    }
    .top-nav-bar {
        flex-direction: column;
        gap: 8px;
        padding: 10px 14px;
    }
    .top-nav-pills {
        width: 100%;
        justify-content: center;
    }
    .nav-pill {
        padding: 6px 14px;
        font-size: 0.72rem;
    }
}

/* ===== Responsive: Tablet ===== */
@media (max-width: 768px) {
    .panel-left, .panel-right {
        min-width: 100%;
    }
}

/* ===== Responsive: Laptop ===== */
@media (max-width: 1024px) {
    .panel-left {
        min-width: 100%;
    }
    .panel-right {
        min-width: 100%;
    }
}

/* ===== Responsive: Desktop ===== */
@media (min-width: 1440px) {
    .gradio-container {
        max-width: 1440px;
    }
}
"""

# ──────────────────────────────────────────────
# Gradio Theme Object
# ──────────────────────────────────────────────

def create_theme() -> gr.themes.Base:
    """Create the ProofyX Gradio theme with Deep Indigo palette."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#eef2ff", c100="#e0e7ff", c200="#c7d2fe", c300="#a5b4fc",
            c400="#818cf8", c500="#6366F1", c600="#4f46e5", c700="#4338ca",
            c800="#3730a3", c900="#312e81", c950="#1e1b4b",
        ),
        secondary_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1",
            c400="#94a3b8", c500="#64748b", c600="#475569", c700="#334155",
            c800="#1e293b", c900="#0f172a", c950="#020617",
        ),
        neutral_hue=gr.themes.Color(
            c50="#fafafa", c100="#f4f4f5", c200="#e4e4e7", c300="#d4d4d8",
            c400="#a1a1aa", c500="#71717a", c600="#52525b", c700="#3f3f46",
            c800="#27272a", c900="#18181b", c950="#09090b",
        ),
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#09090B",
        body_background_fill_dark="#09090B",
        block_background_fill="#18181B",
        block_background_fill_dark="#18181B",
        block_border_width="1px",
        block_border_color="rgba(255,255,255,0.06)",
        block_border_color_dark="rgba(255,255,255,0.06)",
        block_radius="16px",
        block_shadow="none",
        button_primary_background_fill="#6366F1",
        button_primary_background_fill_dark="#6366F1",
        button_primary_text_color="white",
        input_border_color="rgba(255,255,255,0.06)",
        input_border_color_dark="rgba(255,255,255,0.06)",
        input_background_fill="rgba(255,255,255,0.03)",
        input_background_fill_dark="rgba(255,255,255,0.03)",
        input_radius="8px",
        body_text_color="#FAFAFA",
        body_text_color_dark="#FAFAFA",
        body_text_color_subdued="#A1A1AA",
        body_text_color_subdued_dark="#A1A1AA",
    )
