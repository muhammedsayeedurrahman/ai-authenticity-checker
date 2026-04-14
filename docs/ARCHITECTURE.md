# ProofyX Architecture: FastAPI + Gradio

*Version 2.0 | April 2026*

---

## Overview

ProofyX uses a **FastAPI + Gradio mounted together** architecture. FastAPI serves as the real backend (API-first), while Gradio provides the web dashboard UI. Both share the same core detection pipeline.

```
                    ┌─────────────────────────────────┐
                    │        FastAPI Server            │
                    │        (main.py)                 │
                    │                                  │
  Browser ─────────┤  /ui     → Gradio Dashboard      │
  WhatsApp Bot ────┤  /api/v1 → REST API Endpoints    │
  Mobile App ──────┤  /docs   → Swagger (auto-gen)    │
  Batch Script ────┤                                  │
                    │  ┌──────────────────────────┐   │
                    │  │   core/pipeline.py        │   │
                    │  │   (shared detection logic) │   │
                    │  └──────────────────────────┘   │
                    └─────────────────────────────────┘
```

---

## Directory Structure

```
authen_check/
├── main.py                     # FastAPI app entry point + Gradio mount
│
├── core/                       # Framework-agnostic detection pipeline
│   ├── __init__.py
│   ├── pipeline.py             # analyze_image(), analyze_video(), analyze_audio()
│   ├── models.py               # Model loading singleton, inference wrappers
│   ├── reports.py              # PDF/HTML forensic report generation
│   └── metadata.py             # EXIF extraction, file analysis
│
├── api/                        # FastAPI REST API layer
│   ├── __init__.py
│   ├── routes.py               # REST endpoints (/analyze/image, /analyze/video, etc.)
│   ├── schemas.py              # Pydantic request/response models
│   └── auth.py                 # API key authentication (future)
│
├── ui/                         # Gradio UI layer
│   ├── __init__.py
│   ├── gradio_app.py           # Gradio Blocks UI (sidebar, pages, events)
│   ├── components.py           # HTML generators (gauge, bars, verdict, radar)
│   └── theme.py                # CSS variables, theme config, JS injections
│
├── db/                         # Persistence layer
│   ├── __init__.py
│   └── history.py              # SQLite analysis history
│
├── core_models/                # Model architecture definitions (unchanged)
│   ├── corefakenet.py          # CorefakeNet unified hybrid CNN
│   ├── dinov2_auth_model.py    # DINOv2 fine-tuned
│   ├── efficientnet_auth_model.py
│   ├── efficientnet_texture.py # EfficientNet-B4 texture
│   ├── face_deepfake_model.py  # ResNet50 face
│   ├── frequency_cnn.py        # Frequency CNN + FFT preprocessing
│   └── fusion_mlp.py           # Learned fusion + temperature calibration
│
├── pipeline/                   # Analysis orchestration (unchanged)
│   ├── video_analyzer.py       # VideoAnalyzer, extract_frames, FrequencyAnalyzer
│   ├── audio_analyzer.py       # AudioAnalyzer
│   └── face_gate.py            # Face presence detection
│
├── training/                   # Training scripts (unchanged)
│   ├── train_all.py            # Full training pipeline
│   ├── train_corefakenet.py
│   ├── dataset_portraits.py
│   └── ...
│
├── utils/                      # Utilities (unchanged)
│   ├── explainability.py       # Risk explanation generation
│   └── gradcam.py              # GradCAM heatmap generation
│
├── models/                     # Trained model weights (.pth files)
├── assets/                     # Logo, static assets
└── docs/                       # Documentation
    ├── RESEARCH_REPORT.md
    ├── ARCHITECTURE.md         # This file
    └── IMPLEMENTATION_PLAN.md
```

---

## Core Pipeline Contract

All detection functions return **plain Python dicts** with no UI framework dependencies.

### Image Analysis

```python
# core/pipeline.py

def analyze_image(image_pil: PIL.Image, mode: str = "ensemble") -> dict:
    """
    Analyze a single image for deepfake indicators.

    Args:
        image_pil: PIL Image object
        mode: "ensemble" (7-model) or "fast" (CorefakeNet)

    Returns:
        {
            "risk_score": float,              # 0.0 to 1.0
            "risk_percent": float,            # 0.0 to 100.0
            "verdict": str,                   # "LIKELY MANIPULATED" / "POSSIBLY MANIPULATED"
                                              # / "UNCERTAIN" / "LIKELY AUTHENTIC"
            "confidence": str,                # "HIGH" / "MEDIUM" / "LOW"
            "model_agreement": str,           # "5/7 models detect manipulation"
            "model_scores": {
                "vit": float,
                "texture": float,
                "frequency": float,
                "face": float,
                "dino": float,
                "efficientnet": float,
                "forensic": float,
            },
            "fusion_mode": str,               # "learned" / "weighted_avg" / "attention"
            "face_detected": bool,
            "face_aligned": bool,
            "gradcam_image": PIL.Image | None, # PIL Image of heatmap overlay
            "original_image": PIL.Image,       # Original input for display
            "models_used": int,
            "model_versions": dict,            # {"corefakenet": "epoch9", ...}
            "processing_time_ms": float,
            "metadata": {
                "format": str,
                "dimensions": [int, int],
                "file_size_bytes": int,
                "exif": dict | None,
                "has_c2pa": bool,
            },
        }
    """
```

### Video Analysis

```python
def analyze_video(video_path: str, fps: float = 4.0,
                  aggregation: str = "weighted_avg") -> dict:
    """
    Returns:
        {
            "risk_score": float,
            "risk_percent": float,
            "verdict": str,
            "confidence": str,
            "prediction": str,               # "FAKE" / "REAL"
            "total_frames_analyzed": int,
            "fake_frames": int,
            "real_frames": int,
            "faces_detected_in_frames": int,
            "frame_results": [                # Per-frame breakdown
                {
                    "frame_index": int,
                    "timestamp": float,
                    "risk_score": float,
                    "has_face": bool,
                    "model_scores": dict,
                },
            ],
            "temporal_analysis": {
                "score_variance": float,
                "max_frame_jump": float,
                "significant_jumps": int,
                "risk_timeline": [float],     # For chart rendering
            },
            "gradcam_video_path": str | None,
            "video_info": {
                "duration_sec": float,
                "width": int,
                "height": int,
                "fps": float,
            },
            "processing_time_ms": float,
        }
    """
```

### Audio Analysis

```python
def analyze_audio(audio_path: str) -> dict:
    """
    Returns:
        {
            "risk_score": float,              # P(fake)
            "authenticity_score": float,       # 0-100, inverted risk
            "verdict": str,
            "confidence": str,
            "manipulation_type": str,
            "evidence": [str],
            "suspicious_timestamps": [float],
            "segment_results": [
                {
                    "start_time": float,
                    "end_time": float,
                    "fake_probability": float,
                    "real_probability": float,
                },
            ],
            "duration_sec": float,
            "segments_analyzed": int,
            "processing_time_ms": float,
        }
    """
```

### Multimodal Fusion

```python
def analyze_multimodal(image: PIL.Image | None = None,
                       video_path: str | None = None,
                       audio_path: str | None = None) -> dict:
    """
    Returns:
        {
            "risk_score": float,
            "risk_percent": float,
            "verdict": str,
            "confidence": str,
            "media_types": [str],             # ["image", "video", "audio"]
            "modality_scores": {
                "image": float | None,
                "video": float | None,
                "audio": float | None,
            },
            "fusion_weights": dict,
            "explanation": str,
            "processing_time_ms": float,
        }
    """
```

---

## API Layer

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/analyze/image` | Analyze uploaded image |
| `POST` | `/api/v1/analyze/video` | Analyze uploaded video |
| `POST` | `/api/v1/analyze/audio` | Analyze uploaded audio |
| `POST` | `/api/v1/analyze/multimodal` | Analyze multiple media types |
| `POST` | `/api/v1/analyze/url` | Analyze media from URL |
| `GET`  | `/api/v1/history` | List past analyses |
| `GET`  | `/api/v1/history/{id}` | Get specific analysis |
| `GET`  | `/api/v1/history/{id}/report` | Download PDF report |
| `GET`  | `/api/v1/models/status` | List loaded models and status |
| `GET`  | `/api/v1/health` | Health check |

### Response Schema (Pydantic)

```python
# api/schemas.py

class ModelScore(BaseModel):
    name: str
    score: float
    confidence: str

class AnalysisResult(BaseModel):
    id: str                           # UUID for this analysis
    timestamp: datetime
    risk_score: float
    risk_percent: float
    verdict: str
    confidence: str
    model_agreement: str
    model_scores: dict[str, float]
    face_detected: bool
    models_used: int
    processing_time_ms: float
    media_type: str
    metadata: dict

class AnalysisResponse(BaseModel):
    success: bool
    data: AnalysisResult | None
    error: str | None
```

---

## UI Layer (Gradio)

### Page Structure

The Gradio UI simulates multi-page navigation using a sidebar with visibility toggling:

```
┌────────┬──────────────────────────────────────────────┐
│ SIDEBAR│  HEADER: "Ingestion Terminal" | SYSTEM ACTIVE │
│        ├────────────────────────────┬─────────────────┤
│ [Scan] │                            │ Detection       │
│ Anlysis│   Upload Zone              │ Modules Active  │
│ Histor │   (drag & drop)            │ ✓ ViT           │
│ Settngs│                            │ ✓ EfficientNet  │
│        │                            │ ✓ Face CNN      │
│        │   [Initialize Scan]        │ ✓ Frequency     │
│        ├────────────────────────────┤ ✓ CorefakeNet   │
│        │                            ├─────────────────┤
│        │   DETECTION RESULTS        │ RESULTS         │
│        │   ┌──────┬───────┬──────┐  │ [Gauge]         │
│        │   │Gauge │Verdict│Scores│  │ [Verdict]       │
│        │   └──────┴───────┴──────┘  │ [Score Bars]    │
│        │                            │ [Radar Chart]   │
└────────┴────────────────────────────┴─────────────────┘
```

### Key Components

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Stars background | `gr.HTML` + `js_on_load` | Animated starfield |
| Sidebar navigation | `gr.Sidebar` or `gr.Column` + CSS | Page switching |
| System status | `gr.HTML` + `html_template` | Model status display |
| Upload zone | `gr.HTML` + `@children` wrapping `gr.File` | Custom drag-drop |
| Detection modules | `gr.HTML` + `html_template` | Module checklist |
| Risk gauge | SVG via `gr.HTML` | Animated circular gauge |
| Score bars | HTML via `gr.HTML` | Per-model breakdown |
| Radar chart | Chart.js via `gr.HTML` + `head` | Visual score comparison |
| Verdict card | HTML via `gr.HTML` | Color-coded verdict |
| Timeline chart | Chart.js via `gr.HTML` | Video temporal analysis |

---

## Data Flow

### Single Image Analysis

```
User uploads image
    │
    ▼
ui/gradio_app.py: analyze_btn.click()
    │
    ▼
core/pipeline.py: analyze_image(pil_image, mode)
    │
    ├──► core/models.py: get_model_scores(image)
    │       ├──► ViT inference
    │       ├──► EfficientNet-B4 texture inference
    │       ├──► Frequency CNN inference
    │       ├──► DINOv2 inference
    │       ├──► EfficientNet auth inference
    │       ├──► Face model inference
    │       └──► Forensic analysis (heuristic)
    │
    ├──► core/models.py: fuse_scores(scores, mode)
    │       ├──► FusionMLP (learned) OR
    │       └──► CorefakeNet (fast)
    │
    ├──► utils/gradcam.py: generate_gradcam_image()
    │
    ├──► core/metadata.py: extract_metadata()
    │
    └──► Returns plain dict
            │
            ▼
    ui/components.py: render results as HTML
    db/history.py: save to SQLite (async)
            │
            ▼
    Gradio updates UI components
```

### API Request

```
Client sends POST /api/v1/analyze/image
    │
    ▼
api/routes.py: validate request (Pydantic)
    │
    ▼
core/pipeline.py: analyze_image(pil_image, mode)
    │  (same pipeline as Gradio)
    ▼
api/routes.py: format as JSON response
    │
    ▼
Client receives AnalysisResponse JSON
```

---

## Model Loading

Models are loaded once at startup and shared across all requests:

```python
# core/models.py

class ModelRegistry:
    """Singleton that loads all models once and provides inference methods."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.models = {}
        self.loaded = []
        self.missing = []
        self._load_all()

    def _load_all(self):
        self._try_load("dino", DINOv2AuthModel, "dinov2_auth_model.pth")
        self._try_load("efficientnet", EfficientNetAuthModel, "efficientnet_auth_model.pth")
        self._try_load("face", FaceDeepfakeModel, "image_face_model.pth")
        self._try_load("texture", EfficientNetTexture, "efficient.pth")
        self._try_load("frequency", FrequencyCNN, "frequency.pth")
        self._try_load("fusion", FusionMLP, "fusion_mlp.pth", n_inputs=4)
        self._try_load("corefakenet", CorefakeNet, "corefakenet.pth")
        self._load_vit()
        self._load_audio()

    def get_status(self) -> dict:
        return {
            "loaded": self.loaded,
            "missing": self.missing,
            "total": len(self.loaded),
            "corefakenet_ready": "corefakenet" in self.models,
        }

# Global singleton
registry = ModelRegistry()
```

---

## Database Schema (SQLite)

```sql
CREATE TABLE analyses (
    id TEXT PRIMARY KEY,           -- UUID
    timestamp DATETIME NOT NULL,
    media_type TEXT NOT NULL,       -- "image" / "video" / "audio" / "multimodal"
    risk_score REAL NOT NULL,
    verdict TEXT NOT NULL,
    confidence TEXT NOT NULL,
    model_scores TEXT NOT NULL,     -- JSON blob
    face_detected BOOLEAN,
    models_used INTEGER,
    processing_time_ms REAL,
    file_name TEXT,
    file_size_bytes INTEGER,
    metadata TEXT,                  -- JSON blob (EXIF, etc.)
    gradcam_path TEXT,             -- Path to saved heatmap image
    report_path TEXT               -- Path to generated PDF
);

CREATE INDEX idx_analyses_timestamp ON analyses(timestamp DESC);
CREATE INDEX idx_analyses_media_type ON analyses(media_type);
CREATE INDEX idx_analyses_verdict ON analyses(verdict);
```

---

## Future Integrations

### WhatsApp Bot (Phase 4)

```
authen_check/
└── bots/
    └── whatsapp.py      # Twilio webhook
        # POST /webhook → download media → pipeline.analyze_image() → reply
```

### ONNX Export (Phase 3)

```
authen_check/
└── core/
    └── onnx_inference.py  # ONNX Runtime wrapper
        # Same interface as models.py but uses .onnx files
```

### Batch Processing (Phase 3)

```python
# api/routes.py
@app.post("/api/v1/analyze/batch")
async def batch_analyze(files: list[UploadFile]):
    results = [pipeline.analyze_image(open_image(f)) for f in files]
    return {"results": results}
```
