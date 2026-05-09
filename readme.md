# ProofyX

AI-powered multimodal deepfake and media manipulation detection.

ProofyX combines 7+ ML models (ViT, EfficientNet-B4, DINOv2, FrequencyCNN, CorefakeNet, FusionMLP, Audio CNN) with GradCAM explainability and cross-modal fusion to detect manipulated images, videos, and audio.

## Features

- **Multi-model ensemble** -- 7 specialized models with learned fusion
- **CorefakeNet fast mode** -- single unified model with 5 attention-fused heads
- **GradCAM explainability** -- visual heatmaps showing manipulation regions
- **Multimodal analysis** -- image + video + audio cross-modal fusion
- **EXIF forensics** -- metadata anomaly detection
- **REST API** -- OpenAPI-documented endpoints with API key auth
- **Web dashboard** -- Gradio-based UI with 3D neural mesh background

## Quick Start

### Prerequisites

- Python 3.10+
- Model weights in `models/` directory (see [Model Setup](#model-setup))

### Installation

```bash
# Clone
git clone https://github.com/your-org/proofyx.git
cd proofyx

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### Run

```bash
# Start the server
python main.py

# Or with uvicorn directly
uvicorn main:app --host 127.0.0.1 --port 7861
```

Open:
- **Dashboard**: http://127.0.0.1:7861/ui
- **API docs**: http://127.0.0.1:7861/docs
- **REST API**: http://127.0.0.1:7861/api/v1

### Docker

```bash
# Build and run
docker compose up --build

# Or standalone
docker build -t proofyx .
docker run -p 7861:7861 --env-file .env -v ./models:/app/models:ro proofyx
```

## Model Setup

Place trained model weights in the `models/` directory:

| File | Model | Required |
|------|-------|----------|
| `efficientnet_auth_model.pth` | EfficientNet-B4 authenticity | Optional |
| `dinov2_auth_model.pth` | DINOv2 authenticity | Optional |
| `image_face_model.pth` | Face deepfake detector | Optional |
| `efficient.pth` | EfficientNet texture | Optional |
| `frequency.pth` | Frequency domain CNN | Optional |
| `fusion_mlp.pth` | Learned fusion MLP | Optional |
| `corefakenet.pth` | CorefakeNet unified model | Optional |

The ViT model is downloaded automatically from HuggingFace on first run.

ProofyX runs in degraded mode if models are missing -- available models are used, missing ones are skipped.

## API Usage

### Analyze an Image

```bash
curl -X POST http://127.0.0.1:7861/api/v1/analyze/image \
  -H "X-API-Key: your-key" \
  -F "file=@photo.jpg" \
  -F "mode=ensemble"
```

### Analyze a Video

```bash
curl -X POST http://127.0.0.1:7861/api/v1/analyze/video \
  -H "X-API-Key: your-key" \
  -F "file=@video.mp4" \
  -F "fps=4"
```

### Health Check

```bash
curl http://127.0.0.1:7861/api/v1/health
```

All responses follow the envelope pattern:
```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

## Architecture

```
proofyx/
├── main.py                  # FastAPI + Gradio entry point
├── api/
│   ├── routes.py            # REST API endpoints
│   └── schemas.py           # Pydantic request/response models
├── core/
│   ├── config.py            # Configuration loader
│   ├── pipeline.py          # ML inference pipeline
│   ├── secrets.py           # API key pool management
│   └── types.py             # Domain types and enums
├── core_models/
│   ├── corefakenet.py       # CorefakeNet unified architecture
│   ├── efficientnet_*.py    # EfficientNet variants
│   ├── dinov2_auth_model.py # DINOv2 model
│   ├── frequency_cnn.py     # Frequency domain CNN
│   └── fusion_mlp.py        # Learned fusion MLP
├── db/
│   └── history.py           # SQLite analysis history
├── pipeline/
│   ├── video_analyzer.py    # Video frame analysis
│   └── audio_analyzer.py    # Audio analysis
├── ui/
│   ├── gradio_app.py        # Gradio UI layout
│   ├── theme.py             # Design system and CSS
│   └── components.py        # HTML component generators
├── utils/
│   ├── gradcam.py           # GradCAM heatmap generation
│   └── explainability.py    # Risk explanation engine
├── configs/
│   └── models.json          # Model registry configuration
└── models/                  # Model weights (gitignored)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROOFYX_API_KEY_1` | API authentication key | _(none -- dev mode)_ |
| `PROOFYX_API_KEY_2` | Fallback API key | _(none)_ |
| `HF_TOKEN_1` | HuggingFace Hub token | _(none)_ |
| `CORS_ORIGINS` | Comma-separated allowed origins | `localhost:7861` |
| `PROOFYX_TIMEOUT_IMAGE` | Image analysis timeout (seconds) | `60` |
| `PROOFYX_TIMEOUT_VIDEO` | Video analysis timeout (seconds) | `180` |
| `PROOFYX_TIMEOUT_AUDIO` | Audio analysis timeout (seconds) | `90` |
| `PROOFYX_TIMEOUT_MULTIMODAL` | Multimodal analysis timeout (seconds) | `300` |

See `.env.example` for the full list.

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Lint
ruff check .

# Type check
mypy core/ api/ --ignore-missing-imports
```

## License

Proprietary. All rights reserved.
