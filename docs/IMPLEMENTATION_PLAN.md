# ProofyX Implementation Plan

*Version 2.0 | April 2026*

---

## Goal

Transform ProofyX from a Gradio prototype into an API-first product with a professional cybersecurity-themed UI, while keeping all existing detection capabilities intact.

---

## Phase 1: UI Overhaul (3-4 days)

Match the target cybersecurity dashboard design within Gradio.

### 1.1 Sidebar Navigation (2 hrs)

Replace horizontal tabs with `gr.Sidebar`:

```python
with gr.Sidebar(position="left", width=260, elem_id="nav-sidebar"):
    gr.HTML(logo_html)
    scan_btn = gr.Button("LIVE SCAN", elem_classes=["nav-btn", "active"])
    analysis_btn = gr.Button("Analysis", elem_classes=["nav-btn"])
    history_btn = gr.Button("History", elem_classes=["nav-btn"])
    settings_btn = gr.Button("Settings", elem_classes=["nav-btn"])
```

Simulate page routing by toggling `gr.Column` visibility on button clicks.

### 1.2 Stars Background (30 min)

Add animated starfield via `gr.HTML` with `js_on_load`:

```python
gr.HTML(
    html_template="<div class='stars-container'></div>",
    js_on_load="/* Generate 150 animated star divs */",
)
```

### 1.3 System Status Header (30 min)

Add "SYSTEM ACTIVE" + session ID header:

```python
gr.HTML(
    value={"models": len(loaded_models), "session": generate_session_id()},
    html_template="""
    <div class="system-header">
        <h1>Ingestion Terminal</h1>
        <div class="status">
            <span class="pulse-dot"></span> SYSTEM ACTIVE
            <span class="session-id">${value.session}</span>
        </div>
    </div>""",
)
```

### 1.4 Custom Upload Zone (1 hr)

Wrap `gr.File` in custom HTML with `@children`:

```python
with gr.HTML(
    html_template="""
    <div class="upload-zone">
        <div class="zone-icon">&#128274;</div>
        <h3>Initialize ProofyX Scan</h3>
        <p>Drag & drop image, video, or audio for AI-based authenticity detection.</p>
        @children
    </div>""",
):
    file_input = gr.File(file_types=["image", "video", "audio"])
```

### 1.5 Detection Modules Panel (1 hr)

Show loaded model status with green/gray indicators:

```python
modules_panel = gr.HTML(
    value=[{"name": m, "status": "active"} for m in loaded_models],
    html_template="/* Templated module list with status dots */",
)
```

### 1.6 Radar Chart for Model Scores (1 hr)

Add Chart.js radar chart alongside score bars:

```python
radar = gr.HTML(
    value=model_scores_dict,
    html_template="<canvas id='radar'></canvas>",
    js_on_load="/* Chart.js radar initialization */",
    head='<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>',
)
```

### 1.7 Page Simulation (2 hrs)

Implement page switching via sidebar button clicks toggling column visibility:

```python
# Each "page" is a gr.Column with visible=True/False
scan_page = gr.Column(visible=True)
analysis_page = gr.Column(visible=False)
history_page = gr.Column(visible=False)
settings_page = gr.Column(visible=False)

scan_btn.click(fn=show_page, inputs=[], outputs=[scan_page, analysis_page, ...])
```

### 1.8 Scanning Animation (1 hr)

CSS + JS animation during analysis:

```python
scan_progress = gr.HTML(
    html_template="""
    <div class="scan-progress">
        <div class="scan-line"></div>
        <span>SCANNING: ${Math.round(value)}%</span>
    </div>""",
)
```

### 1.9 Model Agreement Indicator (1 hr)

Add "5/7 models detect manipulation" above score bars.

### 1.10 4-Tier Verdict Labels (30 min)

Replace HIGH/MEDIUM/LOW RISK with:
- `> 70%` → "LIKELY MANIPULATED"
- `40-70%` → "POSSIBLY MANIPULATED"
- `30-50%` → "UNCERTAIN"
- `< 30%` → "LIKELY AUTHENTIC"

---

## Phase 2: Product Features (1 week)

### 2.1 EXIF Metadata Extraction (2 hrs)

```python
# core/metadata.py
from PIL.ExifTags import TAGS
import hashlib

def extract_metadata(image_pil, file_path=None):
    exif_data = {}
    raw_exif = image_pil.getexif()
    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, tag_id)
        exif_data[tag_name] = str(value)

    return {
        "format": image_pil.format,
        "dimensions": list(image_pil.size),
        "mode": image_pil.mode,
        "exif": exif_data if exif_data else None,
        "has_exif": bool(exif_data),
        "suspicious_metadata": not bool(exif_data),  # AI images often lack EXIF
        "file_hash": hashlib.sha256(image_pil.tobytes()).hexdigest()[:16],
    }
```

Display in a collapsible "Metadata" panel in the UI.

### 2.2 PDF Forensic Report (1-2 days)

```python
# core/reports.py
from weasyprint import HTML
from jinja2 import Template

REPORT_TEMPLATE = """
<html>
<head><style>/* Professional forensic report CSS */</style></head>
<body>
    <header>
        <img src="logo.png" />
        <h1>ProofyX Forensic Analysis Report</h1>
        <p>Case ID: {{ case_id }} | Date: {{ date }}</p>
    </header>
    <section class="summary">
        <h2>Executive Summary</h2>
        <div class="risk-badge {{ risk_class }}">{{ verdict }}</div>
        <p>Risk Score: {{ risk_percent }}%</p>
    </section>
    <section class="evidence">
        <h2>Visual Evidence</h2>
        <div class="side-by-side">
            <img src="{{ original_b64 }}" />
            <img src="{{ gradcam_b64 }}" />
        </div>
    </section>
    <section class="models">
        <h2>Per-Model Breakdown</h2>
        <table>
            {% for name, score in model_scores.items() %}
            <tr><td>{{ name }}</td><td>{{ (score * 100)|round(1) }}%</td></tr>
            {% endfor %}
        </table>
    </section>
    <section class="metadata">
        <h2>File Metadata</h2>
        <!-- EXIF data table -->
    </section>
    <footer>
        <p>Generated by ProofyX v2.0 | Models: {{ models_used }}</p>
    </footer>
</body>
</html>
"""

def generate_pdf_report(analysis_result: dict, output_path: str) -> str:
    html_content = Template(REPORT_TEMPLATE).render(**analysis_result)
    HTML(string=html_content).write_pdf(output_path)
    return output_path
```

Add "Download Report" button in Gradio UI using `gr.File` for download.

### 2.3 Analysis History with SQLite (1 day)

```python
# db/history.py
import sqlite3
import uuid
from datetime import datetime

class AnalysisHistory:
    def __init__(self, db_path="proofyx_history.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def save(self, result: dict) -> str:
        analysis_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            "INSERT INTO analyses (id, timestamp, media_type, risk_score, ...) VALUES (?, ?, ...)",
            (analysis_id, datetime.now().isoformat(), result["media_type"], ...),
        )
        self.conn.commit()
        return analysis_id

    def get_recent(self, limit=20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]
```

Display in "History" page with a `gr.Dataframe` or custom HTML table.

### 2.4 Processing Time Display (30 min)

Add `time.perf_counter()` around analysis calls, display in results.

### 2.5 Video Temporal Timeline (1 day)

Use Chart.js line chart showing per-frame risk scores over time:

```python
timeline_chart = gr.HTML(
    value={"timestamps": [...], "scores": [...]},
    html_template="<canvas id='timeline'></canvas>",
    js_on_load="""
    new Chart(element.querySelector('#timeline').getContext('2d'), {
        type: 'line',
        data: {
            labels: props.value.timestamps.map(t => t.toFixed(1) + 's'),
            datasets: [{
                label: 'Risk Score',
                data: props.value.scores,
                borderColor: '#00F0FF',
                fill: { target: 'origin', above: 'rgba(236,72,153,0.1)' },
            }]
        },
    });""",
    head='<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>',
)
```

### 2.6 URL-Based Scanning (1 day)

```python
import requests
from PIL import Image
from io import BytesIO

def download_media(url: str) -> tuple[str, bytes]:
    """Download media from URL, return (media_type, content)."""
    response = requests.get(url, timeout=30, stream=True)
    content_type = response.headers.get("content-type", "")
    if "image" in content_type:
        return "image", response.content
    elif "video" in content_type:
        return "video", response.content
    # ... handle audio, YouTube URLs (yt-dlp)
```

Add URL input field alongside file upload.

---

## Phase 3: Architecture Refactor (1-2 weeks)

### 3.1 Extract Core Pipeline (1-2 days)

Move detection logic from `app.py` to `core/pipeline.py`:

1. `app.py:231-439` (analyze_image) → `core/pipeline.py:analyze_image()` returning dict
2. `app.py:443-505` (analyze_image_fast) → same file, `mode="fast"` branch
3. `app.py:517-589` (analyze_video_ui) → `core/pipeline.py:analyze_video()`
4. `app.py:689-735` (analyze_audio_ui) → `core/pipeline.py:analyze_audio()`
5. `app.py:739-818` (analyze_multimodal) → `core/pipeline.py:analyze_multimodal()`

Move model loading from `app.py:118-227` → `core/models.py:ModelRegistry`

### 3.2 FastAPI + Gradio Mount (1 day)

```python
# main.py
from fastapi import FastAPI
from api.routes import router
from ui.gradio_app import create_gradio_app
import gradio as gr

app = FastAPI(title="ProofyX API", version="2.0")
app.include_router(router, prefix="/api/v1")

gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7861)
```

### 3.3 Pydantic Schemas (1 day)

Define request/response models in `api/schemas.py` for type safety and auto-documentation.

### 3.4 REST API Endpoints (1 day)

```python
# api/routes.py
from fastapi import APIRouter, UploadFile, File
from core.pipeline import analyze_image, analyze_video
from api.schemas import AnalysisResponse

router = APIRouter()

@router.post("/analyze/image", response_model=AnalysisResponse)
async def api_analyze_image(file: UploadFile = File(...), mode: str = "ensemble"):
    image = Image.open(file.file)
    result = analyze_image(image, mode=mode)
    return AnalysisResponse(success=True, data=result)

@router.get("/models/status")
async def models_status():
    return registry.get_status()

@router.get("/health")
async def health():
    return {"status": "active", "models_loaded": len(registry.loaded)}
```

### 3.5 ONNX Export (1-2 days)

Start with simple models, defer CorefakeNet (FFT compatibility issue):

```python
# training/export_onnx.py
import torch

# Export FusionMLP (trivial, no exotic ops)
model = FusionMLP(n_inputs=4)
model.load_state_dict(torch.load("models/fusion_mlp.pth"))
model.eval()
dummy = torch.randn(1, 4)
torch.onnx.export(model, dummy, "models/fusion_mlp.onnx", opset_version=17)
```

---

## Phase 4: External Integrations (2-3 weeks)

### 4.1 WhatsApp Bot (2-3 days)

```python
# bots/whatsapp.py
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from core.pipeline import analyze_image

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    media_url = request.form.get("MediaUrl0")
    if media_url:
        image = download_image(media_url)
        result = analyze_image(image, mode="fast")  # Fast mode for latency
        reply_text = f"Risk: {result['risk_percent']:.0f}% - {result['verdict']}"
    else:
        reply_text = "Send an image or video to analyze."

    resp = MessagingResponse()
    resp.message(reply_text)
    return str(resp)
```

### 4.2 Google Vision Reverse Search (1 day)

```python
# core/verification.py
from google.cloud import vision

def reverse_image_search(image_bytes: bytes) -> dict:
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.web_detection(image=image)
    web = response.web_detection

    return {
        "full_matches": len(web.full_matching_images),
        "partial_matches": len(web.partial_matching_images),
        "similar_images": len(web.visually_similar_images),
        "pages_with_image": len(web.pages_with_matching_images),
        "web_entities": [
            {"description": e.description, "score": e.score}
            for e in web.web_entities[:5]
        ],
        "previously_seen_online": len(web.full_matching_images) > 0,
    }
```

### 4.3 C2PA Content Credentials (1 day)

Check for content provenance metadata using the `c2pa-python` library.

### 4.4 Email/SMS Alerts (1 day)

```python
# core/alerts.py
import smtplib
from email.mime.text import MIMEText

def send_alert(result: dict, recipient: str):
    if result["risk_score"] > 0.7:
        msg = MIMEText(f"HIGH RISK detected: {result['verdict']}")
        msg["Subject"] = f"ProofyX Alert: {result['verdict']}"
        # ... send via SMTP or Twilio SMS
```

### 4.5 MLflow Tracking (2 hrs)

Add to training scripts:

```python
import mlflow

with mlflow.start_run(run_name="corefakenet-v2"):
    mlflow.log_params({"batch_size": 4, "lr": 2e-4})
    for epoch in range(num_epochs):
        mlflow.log_metrics({"train_loss": loss, "val_acc": acc}, step=epoch)
    mlflow.pytorch.log_model(model, "corefakenet")
```

---

## Migration Path from Current Code

### What Changes

| Current Location | New Location | Change Type |
|-----------------|-------------|-------------|
| `app.py:231-439` (analyze_image) | `core/pipeline.py` | Move + return dict |
| `app.py:443-505` (analyze_image_fast) | `core/pipeline.py` | Move + return dict |
| `app.py:517-589` (analyze_video_ui) | `core/pipeline.py` | Move + return dict |
| `app.py:689-735` (analyze_audio_ui) | `core/pipeline.py` | Move + return dict |
| `app.py:739-818` (analyze_multimodal) | `core/pipeline.py` | Move + return dict |
| `app.py:118-227` (model loading) | `core/models.py` | Move to ModelRegistry |
| `app.py:1167-1431` (HTML generators) | `ui/components.py` | Move unchanged |
| `app.py:1434-1709` (Gradio Blocks) | `ui/gradio_app.py` | Move + redesign |
| `app.py:1125-1164` (theme) | `ui/theme.py` | Move unchanged |
| `app.py:825-1110` (CSS) | `ui/theme.py` | Move + enhance |

### What Stays Unchanged

- `core_models/` -- All model architecture definitions
- `pipeline/` -- VideoAnalyzer, AudioAnalyzer, face_gate
- `training/` -- All training scripts
- `utils/` -- explainability.py, gradcam.py
- `models/` -- All .pth weight files

### Detection Logic Does NOT Change

The refactor moves functions between files and changes return types from HTML strings to dicts. **Zero changes to model inference, score calculation, fusion, or GradCAM generation.**

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: UI Overhaul | 3-4 days | Cybersecurity dashboard matching target design |
| Phase 2: Product Features | 5-7 days | EXIF, PDF reports, history, timeline, URL input |
| Phase 3: Architecture | 7-10 days | FastAPI + Gradio, REST API, Pydantic schemas |
| Phase 4: Integrations | 10-15 days | WhatsApp bot, reverse search, C2PA, alerts |
| **Total** | **~5-6 weeks** | **Full Image 2 architecture (core features)** |
