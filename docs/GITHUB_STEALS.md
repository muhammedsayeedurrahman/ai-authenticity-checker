# ProofyX: What to Steal from Open-Source — Code-Level Analysis

*Generated: 2026-04-14 | Repos Analyzed: 8 | Source Code Reviewed: 50+ files*

---

## Executive Summary

We deep-dived into 8 GitHub repositories, **reading actual source code** (not just READMEs), to identify exactly which patterns, architectures, and code snippets ProofyX should adopt. Below is a prioritized master list of 30+ stealable patterns organized by impact and effort.

**Top 5 highest-ROI steals:**
1. EXIF Metadata Forensics — free signal, zero training, 10ms per image
2. CLIP Linear Probe (UniversalFakeDetect) — 93.8% mAP with 769 trainable params
3. Standardized PredictionResult type — permanently fixes P(real)/P(fake) inversion bug
4. Lazy Loading Base Class — halves startup time and RAM usage
5. Jinja2+WeasyPrint PDF Reports — professional forensic output

---

## Repositories Analyzed

| # | Repository | Stars | What It Does | Files Read |
|---|-----------|-------|-------------|------------|
| 1 | [DeepSafe](https://github.com/siddharthksah/DeepSafe) | ~200 | Microservice ensemble platform | config, sdk/base.py, sdk/types.py, api/main.py, database.py, manifest.py |
| 2 | [deepfake-detection-v4](https://github.com/ameencaslam/deepfake-detection-project-v4) | ~150 | 5-model Swin/EfficientNet pipeline | train_swin.py, feature_visualization.py, video_processor.py, app.py |
| 3 | [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | ~1.5k | NeurIPS 2023, 36 detectors benchmark | registry.py, base_detector.py, metrics/, dataset/, trainer.py, albu.py |
| 4 | [DeepGuard](https://github.com/camilooscargbaptista/deepguard) | ~80 | Offline forensic analysis + reports | analyzer.py, report.py, models.py, templates/report.html, techniques/*.py |
| 5 | [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect) | ~700 | CVPR 2023, CLIP linear probe | networks/trainer.py, models.py |
| 6 | [DeepFake-Adapter](https://github.com/rshaojimmy/DeepFake-Adapter) | ~100 | IJCV 2025, frozen ViT + adapters | architecture code, training scripts |
| 7 | [M2F2-Det](https://github.com/CHELSEA234/M2F2_Det) | ~150 | CVPR 2025 Oral, LLM explainability | multi-stage pipeline, DDVQA training |
| 8 | [PixelProof](https://github.com/mytechnotalent/pixelproof) | ~50 | EXIF forensics + ELA | pixelproof.py, deep_analysis.py, stego.py |

---

## TIER 1: HIGH Impact, LOW Effort (Do This Week)

### 1.1 EXIF Metadata Forensics
**Source:** PixelProof + DeepGuard
**Effort:** 1 day | **Impact:** HIGH | **CPU-Friendly:** YES (no ML)

Every phone/camera embeds EXIF tags. AI-generated images have none. This is a free forensic signal:

```python
def analyze_exif_forensics(image_path):
    """Forensic EXIF analysis. Returns (suspicion_score, findings)."""
    suspicion_score = 0.0
    findings = []

    img = Image.open(image_path)
    exif_data = img.getexif()

    if not exif_data:
        suspicion_score += 0.3
        findings.append("No EXIF metadata found (common in AI-generated images)")
    else:
        # Camera identifiers
        camera_tags = {271: 'Make', 272: 'Model', 37386: 'FocalLength'}
        if not any(tag in exif_data for tag in camera_tags):
            suspicion_score += 0.2
            findings.append("No camera make/model in EXIF")

        # GPS data
        gps_ifd = exif_data.get_ifd(0x8825)
        if not gps_ifd:
            suspicion_score += 0.1
            findings.append("No GPS data")

        # AI software detection
        software = exif_data.get(305, "")
        ai_keywords = ["stable diffusion", "dall-e", "midjourney", "comfyui"]
        if any(kw in software.lower() for kw in ai_keywords):
            suspicion_score += 0.4
            findings.append(f"AI software detected: {software}")

    return min(suspicion_score, 1.0), findings
```

**Integration point:** Add as non-neural signal in `core/pipeline.py`. Runs <10ms. Provides human-readable evidence for reports.

**Caveat:** Easily bypassed by stripping EXIF. Use as supplementary signal, not primary detector.

---

### 1.2 Standardized PredictionResult Type
**Source:** DeepSafe (`sdk/deepsafe_sdk/types.py`)
**Effort:** 0.5 day | **Impact:** HIGH (fixes documented bug)

**The Problem:** ProofyX MEMORY.md documents: *"texture/frequency models output P(real) not P(fake) → inverted scores"*. Every model has different output semantics.

**The Fix:** Every model returns the same schema:

```python
@dataclass(frozen=True)
class PredictionResult:
    model: str          # "corefakenet", "dino", "vit", etc.
    probability: float  # ALWAYS P(fake). 0.0 = real, 1.0 = fake
    prediction: int     # 1 = fake, 0 = real
    class_name: str     # "fake" or "real"
    inference_time: float
```

DeepSafe enforces this via a `make_result()` factory method in the base class:
```python
def make_result(self, probability, threshold=0.5):
    prediction = 1 if probability >= threshold else 0
    return PredictionResult(
        model=self.name, probability=probability,
        prediction=prediction,
        class_name="fake" if prediction == 1 else "real",
        inference_time=0.0,
    )
```

**Where to create:** `core_models/types.py` — imported by every model file.

---

### 1.3 Lazy Loading Base Class with Idle Unloading
**Source:** DeepSafe (`sdk/deepsafe_sdk/base.py`)
**Effort:** 1-2 days | **Impact:** HIGH (halves startup time + RAM)

**The Problem:** ProofyX loads ALL 7+ models at import time. Startup takes forever. RAM usage is huge.

**The Fix:** Models load on first prediction, unload after idle timeout:

```python
class ProofyXModel(ABC):
    def __init__(self, name, weights_path):
        self._model = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self.name = name
        self.weights_path = weights_path

    def safe_predict(self, image):
        self._ensure_loaded()
        start = time.time()
        result = self.predict(image)
        result.inference_time = time.time() - start
        return result

    def _ensure_loaded(self):
        if self._model is not None:
            self._last_used = time.time()
            return
        with self._lock:
            if self._model is not None:  # double-checked locking
                self._last_used = time.time()
                return
            self.load()
            self._last_used = time.time()

    def check_idle_unload(self, timeout=600):
        if self._model and time.time() - self._last_used > timeout:
            with self._lock:
                del self._model
                self._model = None
                gc.collect()

    @abstractmethod
    def load(self): ...
    @abstractmethod
    def predict(self, image) -> PredictionResult: ...
```

**Key patterns from DeepSafe:**
- Thread-safe double-checked locking prevents duplicate loading
- `check_idle_unload()` frees RAM when models haven't been used
- Especially valuable for CorefakeNet "Fast Mode" where the full ensemble sits idle

---

### 1.4 C2PA Content Credentials Reading
**Source:** [c2pa-python](https://github.com/contentauth/c2pa-python)
**Effort:** 0.5 day | **Impact:** MEDIUM | **CPU-Friendly:** YES (no ML)

Images from Adobe Photoshop, Leica cameras, and Adobe Firefly now include C2PA provenance:

```python
import c2pa

def check_c2pa_provenance(image_path):
    try:
        with open(image_path, "rb") as f:
            reader = c2pa.Reader("image/jpeg", f)
            manifest = reader.json()
            return {"has_c2pa": True, "manifest": manifest, "trust_boost": -0.2}
    except Exception:
        return {"has_c2pa": False, "trust_boost": 0.0}
```

Valid C2PA from a trusted camera = strong authenticity signal. Absence means nothing (most platforms strip it).

---

### 1.5 JSON Model Registry
**Source:** DeepSafe (`config/deepsafe_config.json`)
**Effort:** 0.5 day | **Impact:** MEDIUM

**The Problem:** Model names, weights, paths, and enabled status are hardcoded in `app.py` lines 45-52.

**The Fix:** Config-as-code:

```json
{
  "models": {
    "vit": {"path": "models/vit_deepfake.pth", "weight": 0.40, "enabled": true},
    "dino": {"path": "models/dinov2_auth_model.pth", "weight": 0.25, "enabled": true},
    "corefakenet": {"path": "models/corefakenet.pth", "weight": 1.0, "fast_mode": true}
  },
  "ensemble_method": "fusion_mlp",
  "threshold": 0.5,
  "idle_unload_seconds": 600
}
```

Add/remove/disable models by editing one file. Zero code changes.

---

## TIER 2: HIGH Impact, MEDIUM Effort (Do This Month)

### 2.1 CLIP Linear Probe (UniversalFakeDetect)
**Source:** UniversalFakeDetect (CVPR 2023)
**Effort:** 2-3 days | **Impact:** VERY HIGH | **CPU-Friendly:** YES with ViT-B/32

**The Architecture** is shockingly simple — frozen CLIP + one linear layer:

```python
class CLIPModel(nn.Module):
    def __init__(self, name='ViT-L/14', num_classes=1):
        super().__init__()
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.fc = nn.Linear(768, num_classes)  # Only 769 trainable params!

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        return self.fc(features)
```

**Training:** Freeze everything except `fc.weight` and `fc.bias`. BCEWithLogitsLoss. AdamW lr=8e-5.
**Results:** 93.79% accuracy, 98.66% mAP across 19 generative model test sets.

**Why this matters for ProofyX:** CLIP's multimodal pretraining captures artifacts universal to ALL generators. Your current models might fail on unseen generators (Sora, Flux, etc). CLIP-based detection generalizes.

**ProofyX already has** `general_ai/clip_model.py` using ViT-B/32. The key differences to fix:
1. Use `BCEWithLogitsLoss` (more stable) vs current `Sigmoid + BCELoss`
2. Use single linear layer (768→1) vs current two-layer classifier
3. Keep backbone fully frozen (only 769 trainable params)

**Integration:** Add as independent model in ensemble, NOT as CorefakeNet head (would slow fast mode).

---

### 2.2 Feature Map Visualization (Multi-Layer Activation Maps)
**Source:** deepfake-detection-v4 (`feature_visualization.py`)
**Effort:** 2-3 days | **Impact:** HIGH (major explainability gap)

**The Problem:** ProofyX only has GradCAM (single final-layer heatmap). No intermediate layer visualization.

**The Solution:** Forward hooks capture activations at every stage:

```python
class FeatureExtractor:
    def __init__(self, model):
        self.features = {}
        self.hooks = []

    def _hook_fn(self, name, output):
        self.features[name] = output.detach()

    def register(self, model):
        # Hook early, mid, and late layers
        model.backbone_early.register_forward_hook(
            lambda m, i, o: self._hook_fn('early_24x24', o))
        model.backbone_late.register_forward_hook(
            lambda m, i, o: self._hook_fn('late_12x12', o))

class FeatureVisualizer:
    @staticmethod
    def normalize_feature_map(fmap):
        fmap = fmap.squeeze(0).max(0)[0]  # Max across channels
        fmap = fmap.cpu().numpy()
        return (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

    @staticmethod
    def create_heatmap(fmap, size=(380, 380)):
        resized = cv2.resize(fmap, size)
        return cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_INFERNO)
```

**Benefits over GradCAM alone:**
- No gradient computation needed (faster)
- Multi-layer visualization (early, mid, late features)
- Shows what each CorefakeNet head "sees"
- COLORMAP_INFERNO has better perceptual uniformity than current COLORMAP_HOT
- Complementary to GradCAM, not a replacement

---

### 2.3 Jinja2 + WeasyPrint PDF Forensic Reports
**Source:** DeepGuard (`report.py` + `templates/report.html`)
**Effort:** 3-4 days | **Impact:** HIGH (professional output)

**Architecture:**
```python
class ReportGenerator:
    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))

    def generate_html(self, report, image_path=None):
        image_b64 = image_to_base64(Image.open(image_path)) if image_path else None
        template = self.env.get_template("forensic_report.html")
        return template.render(report=report, image_base64=image_b64)

    def generate_pdf(self, report, image_path=None):
        html = self.generate_html(report, image_path)
        return weasyprint.HTML(string=html).write_pdf()
```

**Report sections (from DeepGuard, extended for multimodal):**
1. Header — ProofyX branding, verdict badge, confidence meter
2. Executive Summary — 4-card grid (Media Type, Risk Level, Time, Models Used)
3. Source Media — Base64-embedded image
4. Model-by-Model Analysis — Per-model cards with GradCAM
5. Forensic Techniques — ELA, Noise, Frequency analysis
6. Audio Analysis (conditional) — Spectral features
7. Video Temporal Analysis (conditional) — Timeline chart
8. EXIF Metadata — Table
9. Recommendations
10. Footer — Legal disclaimer

**Key CSS from DeepGuard worth stealing:**
- Dark theme: `background: linear-gradient(135deg, #0a0a0f, #1a1a2e)`
- Verdict badges: colored pills with `border-radius: 50px`
- Confidence meter: gradient bar `linear-gradient(90deg, #10b981, #f59e0b, #ef4444)`

---

### 2.4 Registry + Abstract Detector Pattern
**Source:** DeepfakeBench (`detectors/base_detector.py`, `metrics/registry.py`)
**Effort:** 1-2 days | **Impact:** MEDIUM-HIGH

Self-registering detectors with a standardized contract:

```python
# core_models/registry.py
class Registry:
    def __init__(self):
        self._modules = {}

    def register(self, name=None):
        def _register(cls):
            self._modules[name or cls.__name__] = cls
            return cls
        return _register

    def __getitem__(self, key):
        return self._modules[key]

DETECTOR = Registry()
```

```python
# core_models/base_detector.py
class BaseDetector(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, data_dict, inference=False) -> dict:
        """Must return dict with 'prob' (P(fake)), 'feat', 'cls'."""

    @abc.abstractmethod
    def get_losses(self, data_dict, pred_dict) -> dict:
        """Must return dict with 'overall' key."""

    @abc.abstractmethod
    def get_train_metrics(self, data_dict, pred_dict) -> dict:
        """Must return dict with 'acc', 'auc', 'eer', 'ap'."""
```

```python
# core_models/corefakenet.py
@DETECTOR.register('corefakenet')
class CorefakeNet(BaseDetector):
    ...
```

**Pipeline becomes config-driven:**
```python
model = DETECTOR[config['model_name']](config)
```

Adding a new detector = one file + one YAML config. Zero pipeline changes.

---

### 2.5 Albumentations + JPEG Compression Augmentation
**Source:** DeepfakeBench (`dataset/albu.py`, `dataset/abstract_dataset.py`)
**Effort:** 1 day | **Impact:** HIGH (training robustness)

**Critical missing augmentation in ProofyX:** JPEG compression simulation.

Social media heavily compresses images. Models trained without compression augmentation fail on compressed deepfakes.

```python
import albumentations as A
import cv2

train_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.OneOf([
        A.Resize(380, 380, interpolation=cv2.INTER_AREA),
        A.Resize(380, 380, interpolation=cv2.INTER_CUBIC),
        A.Resize(380, 380, interpolation=cv2.INTER_LINEAR),
    ], p=1),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        A.FancyPCA(),
        A.HueSaturationValue(),
    ], p=0.5),
    # THE KEY ADDITION:
    A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.pytorch.ToTensorV2(),
])
```

**Also from DeepfakeBench — Resize4xAndBack augmentation:**
```python
class Resize4xAndBack(ImageOnlyTransform):
    """Simulates social-media downscale-then-upscale artifacts."""
    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h), interpolation=random.choice([
            cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img
```

---

### 2.6 NaN-Tolerant Ensemble
**Source:** DeepSafe (`api/main.py`, lines 645-741)
**Effort:** 1 day | **Impact:** MEDIUM-HIGH

**The Problem:** If face model returns no result (no face detected), ProofyX takes a completely different code path with different weights.

**The Fix:** Missing model outputs are NaN, handled by an imputer:

```python
scores = {}
for model_name, model in active_models.items():
    try:
        scores[model_name] = model.safe_predict(image).probability
    except Exception:
        scores[model_name] = np.nan  # imputer handles it

feature_vector = [scores.get(col, np.nan) for col in expected_columns]
imputed = SimpleImputer(strategy='mean').fit_transform([feature_vector])
final_score = fusion_model.predict(imputed)
```

Eliminates the `WEIGHTS` vs `WEIGHTS_FACE_BOOSTED` bifurcation in current code.

---

## TIER 3: MEDIUM Impact, MEDIUM Effort (Do Next Month)

### 3.1 Standardized Evaluation Metrics (EER, AP, Video-AUC)
**Source:** DeepfakeBench (`metrics/utils.py`)
**Effort:** 0.5 day | **Impact:** MEDIUM

ProofyX's eval is missing two critical metrics the deepfake community uses:

```python
def compute_deepfake_metrics(y_pred, y_true, video_names=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)

    # Standard
    auc = metrics.auc(fpr, tpr)
    acc = (np.array(y_pred > 0.5, dtype=int) == y_true).mean()

    # MISSING from ProofyX:
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]  # Equal Error Rate
    ap = average_precision_score(y_true, y_pred)              # Average Precision

    result = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    if video_names is not None:
        # Video-level: average frame predictions per video, then compute AUC
        result['video_auc'] = _compute_video_auc(y_pred, y_true, video_names)

    return result
```

---

### 3.2 YAML-Driven Training Configuration
**Source:** DeepfakeBench
**Effort:** 1 day | **Impact:** MEDIUM

Replace hardcoded constants in training scripts with YAML configs:

```yaml
# configs/corefakenet.yaml
model_name: corefakenet
input_size: 380
batch_size: 4
epochs: 30
max_samples: 3000
loss: multi_task_bce
optimizer:
  type: adam
  lr: 0.001
  weight_decay: 0.01
augmentation:
  flip_prob: 0.5
  jpeg_quality_range: [40, 100]
```

Enables experiment reproducibility and hyperparameter sweeps without editing source code.

---

### 3.3 Swin Transformer Architecture
**Source:** deepfake-detection-v4 (`train_swin.py`)
**Effort:** 3-4 days | **Impact:** MEDIUM-HIGH

ProofyX has no Swin Transformer. Key advantages:
- **ImageNet-22K pretraining** (10x more data than ImageNet-1K)
- **Hierarchical shifted-window attention** captures multi-scale features
- **GELU + LayerNorm** throughout (better than ReLU + Linear)

```python
class DeepfakeSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224_in22k',
            pretrained=True, num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
```

**Integration:** Add as standalone model in ensemble, or replace CorefakeNet's ViT-style head.

---

### 3.4 Reusable Trainer Class
**Source:** DeepfakeBench (`trainer/trainer.py`)
**Effort:** 2-3 days | **Impact:** MEDIUM

Currently every training script has its own copy-pasted training loop. Extract into reusable class:

```python
class Trainer:
    def __init__(self, config, model, optimizer, scheduler):
        self.config = config
        self.model = model
        self.best_metric = float('-inf')

    def train_epoch(self, epoch, train_loader, val_loader=None):
        self.model.train()
        for data_dict in train_loader:
            preds = self.model(data_dict)
            losses = self.model.get_losses(data_dict, preds)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        if val_loader:
            return self.evaluate(val_loader)
```

Then every training script becomes 10 lines:
```python
model = DETECTOR[config['model_name']](config)
trainer = Trainer(config, model, optimizer, scheduler)
for epoch in range(config['epochs']):
    trainer.train_epoch(epoch, train_loader, val_loader)
```

---

### 3.5 Weight Manifest with Auto-Download + SHA256
**Source:** DeepSafe (`sdk/deepsafe_sdk/manifest.py`, `weights.py`)
**Effort:** 1 day | **Impact:** MEDIUM

```yaml
# models/manifest.yaml
models:
  - name: corefakenet
    path: models/corefakenet.pth
    sha256: "abc123..."
    url: "https://huggingface.co/proofyx/corefakenet/resolve/main/corefakenet.pth"
  - name: dinov2_auth
    path: models/dinov2_auth_model.pth
    sha256: "def456..."
    url: "https://huggingface.co/proofyx/dinov2/resolve/main/dinov2_auth_model.pth"
```

On startup: auto-download missing models, verify SHA256, re-download on mismatch. Makes `git clone && python main.py` just work.

---

### 3.6 Analysis History with SQLite
**Source:** DeepSafe (`api/database.py`)
**Effort:** 1-2 days | **Impact:** MEDIUM

```python
class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id = Column(String, primary_key=True)  # UUID
    timestamp = Column(DateTime, default=datetime.utcnow)
    media_type = Column(String)
    verdict = Column(String)
    risk_score = Column(Float)
    ensemble_method = Column(String)
    processing_time = Column(Float)
    full_response = Column(Text)  # Full JSON blob
```

Every analysis persisted. Users can view past results. System is auditable.

---

### 3.7 Template-Based Explainability (Inspired by M2F2-Det)
**Source:** M2F2-Det (CVPR 2025 Oral)
**Effort:** 2 days | **Impact:** MEDIUM

M2F2-Det uses a 7B LLM for forensic explanations (not feasible on CPU). But the concept adapts to templates:

```python
def explain_with_evidence(corefakenet_output):
    scores = corefakenet_output['head_scores'][0]
    attn = corefakenet_output['attention_weights'][0]

    findings = []
    if scores[0] > 0.6:  # texture head
        findings.append(
            f"Texture analysis (weight: {attn[0]:.0%}) detected "
            f"photorealistic inconsistencies typical of GAN/diffusion output"
        )
    if scores[1] > 0.6:  # frequency head
        findings.append(
            f"Spectral analysis (weight: {attn[1]:.0%}) found "
            f"frequency-domain anomalies in mid-level features"
        )
    if scores[2] > 0.6:  # artifact head
        findings.append(
            f"Edge artifact detection (weight: {attn[2]:.0%}) identified "
            f"blending seams consistent with face-swapping techniques"
        )
    return findings
```

Uses CorefakeNet's per-head scores + attention weights to generate M2F2-Det-style natural language explanations without any LLM.

---

## TIER 4: NICE-TO-HAVE (Future Backlog)

### 4.1 Bottleneck Adapter Pattern
**Source:** DeepFake-Adapter (IJCV 2025)

Residual bottleneck adapters for regularization:
```python
class BottleneckAdapter(nn.Module):
    def __init__(self, in_dim=1792, bottleneck=64):
        self.down = nn.Linear(in_dim, bottleneck)
        self.up = nn.Linear(bottleneck, in_dim)
    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))  # residual
```
Could improve CorefakeNet head regularization. Not urgent.

### 4.2 Cross-Attention on CNN Spatial Features
**Source:** deepfake-detection-v4 (`train_cross_attention.py`)

Treat CNN spatial features as tokens for self-attention:
```python
features = backbone(x)              # (B, 1792, 12, 12)
tokens = features.view(B, 1792, -1).transpose(1, 2)  # (B, 144, 1792)
attended = cross_attention(tokens)   # Self-attention over spatial positions
pooled = attended.mean(dim=1)        # Global average pool
```
Interesting for CorefakeNet's ViT-style head but adds latency.

### 4.3 Per-Model Video Breakdown + Face Gallery
**Source:** deepfake-detection-v4 (`video_processor.py`, `app.py`)

Show each model's verdict across video frames independently. Display detected faces as a visual grid. UI enhancement, not detection improvement.

### 4.4 Model Scaffolding CLI
**Source:** DeepSafe (`scripts/add_model.py`)

`python scripts/add_model.py --name wavelet_detector` auto-generates model file, updates config. Nice DX but not critical yet.

### 4.5 Docker Containerization
**Source:** DeepSafe (`docker-compose.yml`)

Separate containers for UI and model inference. Enables running models on GPU box while UI runs locally. Defer until scaling is needed.

---

## What ProofyX Already Does Better Than ALL of These

Before stealing anything, acknowledge what we have that they don't:

| ProofyX Advantage | None of the Repos Have This |
|---|---|
| **CorefakeNet unified hybrid** | They all run separate models. We share a backbone across 5 heads → 4.9x faster |
| **Learned FusionMLP** with temperature calibration | They use simple voting or averaging |
| **Temporal video analysis** with sliding-window variance | deepfake-detection-v4 uses simple majority vote per frame |
| **Audio deepfake detection** | Only DeepSecure-AI attempts this, and poorly |
| **GradCAM on 4 architectures** with face-region overlay | deepfake-detection-v4 has feature maps but no face alignment |
| **Frequency domain analysis** (trained FrequencyCNN + FFT heuristic) | None have this |
| **Multi-head attention fusion** with learnable temperature | Most advanced fusion in any open-source project |

---

## Master Priority Table

| # | What to Steal | From | Effort | Impact | CPU? |
|---|---|---|---|---|---|
| 1 | EXIF Metadata Forensics | PixelProof + DeepGuard | 1 day | HIGH | YES |
| 2 | PredictionResult type (fix inversion bug) | DeepSafe | 0.5 day | HIGH | N/A |
| 3 | Lazy Loading + Idle Unload | DeepSafe | 1-2 days | HIGH | N/A |
| 4 | C2PA Reading | c2pa-python | 0.5 day | MEDIUM | YES |
| 5 | JSON Model Registry | DeepSafe | 0.5 day | MEDIUM | N/A |
| 6 | CLIP Linear Probe | UniversalFakeDetect | 2-3 days | VERY HIGH | YES |
| 7 | Feature Map Visualization | deepfake-detection-v4 | 2-3 days | HIGH | N/A |
| 8 | PDF Forensic Reports | DeepGuard + WeasyPrint | 3-4 days | HIGH | N/A |
| 9 | Registry + BaseDetector | DeepfakeBench | 1-2 days | MEDIUM-HIGH | N/A |
| 10 | Albumentations + JPEG augment | DeepfakeBench | 1 day | HIGH | N/A |
| 11 | NaN-tolerant ensemble | DeepSafe | 1 day | MEDIUM-HIGH | N/A |
| 12 | Standardized metrics (EER, AP) | DeepfakeBench | 0.5 day | MEDIUM | N/A |
| 13 | YAML training configs | DeepfakeBench | 1 day | MEDIUM | N/A |
| 14 | Swin Transformer | deepfake-detection-v4 | 3-4 days | MEDIUM-HIGH | Marginal |
| 15 | Reusable Trainer class | DeepfakeBench | 2-3 days | MEDIUM | N/A |
| 16 | Weight manifest + auto-download | DeepSafe | 1 day | MEDIUM | N/A |
| 17 | SQLite analysis history | DeepSafe | 1-2 days | MEDIUM | N/A |
| 18 | Template explainability | M2F2-Det inspired | 2 days | MEDIUM | YES |

**Recommended sprint order:** Items 1-5 in Week 1, Items 6-11 in Week 2-3, Items 12-18 as backlog.

---

## Sources

### Repositories (Source Code Read)
- [DeepSafe](https://github.com/siddharthksah/DeepSafe) — Microservice ensemble platform
- [deepfake-detection-v4](https://github.com/ameencaslam/deepfake-detection-project-v4) — Swin + feature visualization
- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) — NeurIPS 2023, 36 detectors
- [DeepGuard](https://github.com/camilooscargbaptista/deepguard) — Forensic reports
- [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect) — CVPR 2023, CLIP linear probe
- [DeepFake-Adapter](https://github.com/rshaojimmy/DeepFake-Adapter) — IJCV 2025, frozen ViT + adapters
- [M2F2-Det](https://github.com/CHELSEA234/M2F2_Det) — CVPR 2025 Oral, LLM explainability
- [PixelProof](https://github.com/mytechnotalent/pixelproof) — EXIF forensics

### Papers
- [Unlocking Hidden Potential of CLIP in Deepfake Detection](https://arxiv.org/abs/2503.19683) — LNCLIP-DF, 0.03% params fine-tuned
- [DeepFake-Adapter: Dual-Level Adapter for DeepFake Detection](https://arxiv.org/abs/2306.00863) — IJCV 2025
- [M2F2-Det: Multi-Modal and Multi-Face Forgery Detection](https://arxiv.org/abs/2411.13295) — CVPR 2025 Oral
- [Forensic Analysis of Image Metadata](https://www.researchgate.net/publication/394477131) — 89%+ accuracy from EXIF alone

### Libraries
- [c2pa-python](https://pypi.org/project/c2pa-python/) — Content Credentials SDK
- [Albumentations](https://albumentations.ai/) — Fast augmentation library
- [WeasyPrint](https://weasyprint.org/) — HTML/CSS to PDF
- [timm](https://github.com/huggingface/pytorch-image-models) — Swin Transformer models
