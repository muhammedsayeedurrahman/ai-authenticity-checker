# ProofyX Implementation Plan: Research Prototype to Production MVP

> Generated from an independent code review on 2026-05-09.
> This document captures the current state assessment and the prioritized roadmap
> to make ProofyX investor-ready and deployable for paying customers.

---

## 1. Current State Assessment

### What Works

| Area | Status | Details |
|------|--------|---------|
| ML Models | Trained | 7 checkpoints (696 MB), proper preprocessing, novel CorefakeNet architecture |
| REST API | Functional | FastAPI with OpenAPI docs, file validation, rate limiting |
| Web UI | Functional | Gradio 6 dashboard with dark theme, GradCAM overlays |
| Security Basics | Implemented | API key auth, parameterized SQL, magic byte validation, security headers |
| Docker | Ready | Dockerfile + compose with GPU support, non-root user |
| CI/CD | Partial | GitHub Actions: lint, test, docker build |
| Tests | Partial | API integration, pipeline unit, database, secrets tests |

### What Doesn't Work (Blockers to Deployment)

| Issue | Severity | Impact |
|-------|----------|--------|
| No published accuracy benchmarks | CRITICAL | Cannot prove value to investors or customers |
| Single Uvicorn worker | HIGH | One slow request blocks all traffic |
| No request timeouts | HIGH | Adversarial/large inputs can hang the server |
| XSS in `ui/components.py` | HIGH | User-supplied filenames rendered as raw HTML |
| ViT downloaded at runtime from HuggingFace | MEDIUM | Service fails to start if HF is down |
| No monitoring/observability | MEDIUM | Blind in production (no metrics, no alerts) |
| No model versioning | MEDIUM | Cannot roll back, A/B test, or track provenance |
| `core/pipeline.py` is 815 lines | MEDIUM | Merge conflicts, hard to maintain |
| No business logic (billing, users, webhooks) | LOW | Research tool, not a product |

---

## 2. Prioritized Implementation Phases

### Phase 1: Prove the Model (CRITICAL -- Do First)

**Goal:** Answer "what's your accuracy?" with hard numbers.

#### 1.1 Run Benchmarks on Standard Datasets

- [ ] Evaluate all 7 models + CorefakeNet on **FaceForensics++** (c23, c40)
- [ ] Evaluate on **Celeb-DF v2**
- [ ] Evaluate on **WildDeepfake**
- [ ] Evaluate on AI-generated images: **DiffusionDB**, **Midjourney**, **DALL-E 3 samples**
- [ ] Record: ROC-AUC, F1, Precision, Recall, FPR at 1%/5% thresholds
- [ ] Record: per-model scores AND ensemble/CorefakeNet scores

**Files involved:**
- `scripts/run_benchmarks.py` -- extend to output structured JSON + markdown
- `training/evaluate.py` -- ensure it covers all standard datasets
- `evaluation/evaluate_pipeline.py` -- end-to-end pipeline evaluation

#### 1.2 Publish Benchmark Table in README

Add a section like:

```markdown
## Benchmarks

| Dataset | ROC-AUC | F1 | Precision | Recall |
|---------|---------|----|-----------| -------|
| FaceForensics++ (c23) | 0.XX | 0.XX | 0.XX | 0.XX |
| Celeb-DF v2 | 0.XX | 0.XX | 0.XX | 0.XX |
| WildDeepfake | 0.XX | 0.XX | 0.XX | 0.XX |
| AI-Generated (mixed) | 0.XX | 0.XX | 0.XX | 0.XX |

CorefakeNet achieves Xx speedup over 7-model ensemble with Y% accuracy retention.
```

#### 1.3 Adversarial Robustness

- [ ] Run `scripts/eval_adversarial.py` (already exists) against JPEG compression, resize, blur, noise
- [ ] Document degradation curves in benchmarks

**Deliverable:** A README benchmarks section with real numbers. No estimates, no claims without data.

---

### Phase 2: Fix Security and Stability (HIGH -- Do Before Any Public Deployment)

#### 2.1 Fix XSS Vulnerability -- DONE

**File:** `ui/components.py`

- [x] Add `from html import escape` at top
- [x] Escape all user-controlled data rendered in HTML:
  - `file_name` -- uploaded filename
  - `agreement` string
  - EXIF `key`/`val` pairs
  - EXIF `findings` list (both locations)
  - Model `name` fields (score bars + modules panel)
  - `verdict_str` in verdict card
  - `label` in gauge (aria-label + SVG text)
  - `logo_url` in img src (escaped + scheme-validated)
  - `message` in empty state
  - All history table entry fields (id, timestamp, media_type, verdict, file_name)

#### 2.2 Add Request Timeouts

**File:** `api/routes.py`

- [ ] Add configurable timeout (default 60s for images, 120s for video, 60s for audio)
- [ ] Use `asyncio.wait_for()` or `starlette` background task with timeout
- [ ] Return HTTP 504 Gateway Timeout on expiry
- [ ] Add `PROOFYX_TIMEOUT_SECONDS` env var

#### 2.3 Increase Worker Count

**File:** `main.py`

- [ ] Change to `workers=4` (or configurable via `PROOFYX_WORKERS` env var)
- [ ] Add `--preload` flag to share model memory across workers
- [ ] Document GPU memory requirements per worker
- [ ] Consider Gunicorn with Uvicorn workers for production

#### 2.4 Bake Models into Docker Image

**File:** `Dockerfile`

- [ ] Download ViT model during `docker build` (not at runtime)
- [ ] Add `RUN python -c "from transformers import ...; model = ..."` build step
- [ ] Or: document a `models/` volume mount that includes the ViT cache

#### 2.5 Add Model Integrity Checks

**File:** `configs/models.json`

- [ ] Add `sha256` field for each model file
- [ ] Verify checksum on load in `core/pipeline.py`
- [ ] Fail loudly if checksum mismatch (prevents loading tampered weights)

---

### Phase 3: Production Infrastructure (MEDIUM -- Before Paying Customers)

#### 3.1 Monitoring and Observability

- [ ] Add structured JSON logging (replace print-style logs)
- [ ] Integrate Prometheus metrics endpoint (`/metrics`):
  - Request count, latency histograms (P50, P95, P99)
  - Model inference time per model
  - GPU memory utilization
  - Error rates by type
- [ ] Add health check that verifies models are loaded (not just HTTP 200)
- [ ] Set up alerting thresholds (latency > 10s, error rate > 5%)

#### 3.2 Model Versioning

- [ ] Add version field to `configs/models.json` (e.g., `"version": "1.0.0"`)
- [ ] Include version in API response (`model_versions` field)
- [ ] Tag model releases in git (e.g., `model-v1.0.0`)
- [ ] Store training metadata alongside checkpoints:
  - Training date, dataset versions, hyperparameters, final metrics
- [ ] Document rollback procedure

#### 3.3 Refactor Pipeline Monolith

**Current:** `core/pipeline.py` (815 lines)

**Target:**
```
core/
├── pipeline.py          # Orchestrator only (~200 lines)
├── image_analyzer.py    # Image analysis logic
├── video_analyzer.py    # Video analysis logic
├── audio_analyzer.py    # Audio analysis logic
├── multimodal_fusion.py # Cross-modal fusion
└── scoring.py           # Calibration, verdict, risk level
```

- [ ] Extract `analyze_image()` to `core/image_analyzer.py`
- [ ] Extract `analyze_video()` to `core/video_analyzer.py`
- [ ] Extract `analyze_audio()` to `core/audio_analyzer.py`
- [ ] Extract scoring/calibration functions to `core/scoring.py`
- [ ] Keep `pipeline.py` as thin orchestrator

#### 3.4 Async Task Queue

- [ ] Add Celery + Redis (or `arq` for lightweight) for async analysis
- [ ] Return `202 Accepted` with `analysis_id` for long-running video/multimodal
- [ ] Add `GET /api/v1/analysis/{id}/status` polling endpoint
- [ ] Add optional webhook callback on completion

#### 3.5 Reverse Proxy and TLS

- [ ] Add nginx config for production (TLS termination, static file serving)
- [ ] Or: document deployment behind cloud load balancer (ALB, Cloud Run, etc.)
- [ ] Update CORS to use production origins

---

### Phase 4: Product Features (LOW -- Before Revenue)

#### 4.1 User Management

- [ ] JWT-based authentication (or OAuth2 via Auth0/Clerk)
- [ ] User registration, login, API key generation
- [ ] Per-user rate limits and usage tracking
- [ ] Admin dashboard for user management

#### 4.2 Usage Tracking and Billing

- [ ] Track API calls per user (count, media type, processing time)
- [ ] Stripe integration for usage-based billing
- [ ] Free tier limits (e.g., 100 analyses/month)
- [ ] Usage dashboard in Gradio UI

#### 4.3 Batch Processing API

- [ ] `POST /api/v1/analyze/batch` -- accept ZIP or multiple files
- [ ] Return batch job ID, poll for results
- [ ] CSV/JSON export of batch results

#### 4.4 SDK and Client Libraries

- [ ] Python SDK (`pip install proofyx`)
- [ ] JavaScript/TypeScript SDK (`npm install proofyx`)
- [ ] Code examples in README

#### 4.5 Webhook Notifications

- [ ] Register webhook URLs per user
- [ ] POST results on analysis completion
- [ ] Retry with exponential backoff
- [ ] HMAC signature verification

---

## 3. Technical Debt Register

| Item | Location | Priority | Effort |
|------|----------|----------|--------|
| Broad `except Exception` clauses | `core/pipeline.py`, model loaders | MEDIUM | Small |
| Magic numbers (threshold 0.5, temperature 1.2) | Multiple files | LOW | Small |
| No type checking in CI | `.github/workflows/ci.yml` | LOW | Small |
| CI jobs run sequentially | `.github/workflows/ci.yml` | LOW | Small |
| Missing architecture docs | `docs/ARCHITECTURE.md` referenced but absent | LOW | Medium |
| No pip-audit in CI | `.github/workflows/ci.yml` | MEDIUM | Small |
| Audio model underdocumented | `core_models/audio_deepfake_model.py` | LOW | Small |

---

## 4. Competitive Positioning (Needs Resolution)

The deepfake detection space has funded competitors:

| Competitor | Funding | Differentiator |
|------------|---------|----------------|
| Reality Defender | $15M+ | Enterprise API, real-time |
| Sensity AI | $7M+ | Threat intelligence platform |
| Hive Moderation | $120M+ | Multi-modal content moderation |
| Intel FakeCatcher | Internal | Real-time, 96% claimed accuracy |

**ProofyX must articulate its edge.** Candidates:
- **Open-source / self-hosted** -- enterprises that can't send data to third parties
- **Multi-model explainability** -- GradCAM + per-model scores (competitors are black boxes)
- **CorefakeNet speed** -- single model with ensemble-level accuracy
- **On-premise deployment** -- Docker + GPU, no cloud dependency
- **Price** -- undercut enterprise pricing

**Action item:** Write a one-paragraph positioning statement and add it to the README.

---

## 5. Dataset Licensing (Legal Risk)

Training data sourced from 10 HuggingFace datasets. Before commercial use:

- [ ] Audit license of each dataset (CC-BY, CC-BY-NC, research-only, etc.)
- [ ] Remove or replace any non-commercial datasets if selling the product
- [ ] Document data provenance in `docs/DATA_PROVENANCE.md`
- [ ] Consult legal counsel on model weights derived from restricted data

---

## 6. Deployment Checklist (Pre-Launch)

- [ ] Accuracy benchmarks published in README
- [ ] XSS vulnerability fixed
- [ ] Request timeouts configured
- [ ] Multiple workers or task queue
- [ ] All models baked into Docker image (no runtime downloads)
- [ ] Model checksums verified on load
- [ ] HTTPS/TLS configured (reverse proxy or cloud LB)
- [ ] Structured logging enabled
- [ ] Health check verifies model loading
- [ ] CORS origins set to production domains
- [ ] Rate limits tuned for production traffic
- [ ] Error messages sanitized (no stack traces to clients)
- [ ] `pip-audit` clean (no known vulnerabilities)
- [ ] Dataset licenses audited
- [ ] Competitive positioning documented
- [ ] Load testing completed (target: X concurrent users)

---

## 7. Summary

ProofyX has strong ML engineering and decent application architecture. The gap is
between "working prototype" and "deployable product." Phase 1 (benchmarks) is
existential -- without accuracy numbers, nothing else matters. Phase 2
(security/stability) is required before any public-facing deployment. Phases 3-4
build toward revenue.

**Current readiness: ~60% to credible MVP.**
