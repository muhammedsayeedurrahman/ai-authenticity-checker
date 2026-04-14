"""
Evaluate CorefakeNet vs. multi-model ensemble.

Compares on the same held-out validation set:
  - Accuracy, ROC-AUC, Precision, Recall, F1
  - Per-head score analysis for CorefakeNet
  - Inference speed (per image, per 22s video estimate)

Usage:
    python training/eval_corefakenet.py
"""

import sys
import os
import time
import random

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache", "datasets")

import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from training.dataset_portraits import load_portrait_dataset

MODELS_DIR = os.path.join(ROOT_DIR, "models")
EVAL_SAMPLES = 500        # Held-out evaluation set
SPEED_ITERATIONS = 20     # Warm-up + timed iterations for speed benchmark


# ================================================================
#  Load Models
# ================================================================

def load_corefakenet(device):
    """Load trained CorefakeNet."""
    from core_models.corefakenet import CorefakeNet

    model_path = os.path.join(MODELS_DIR, "corefakenet.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Train first.")
        return None

    model = CorefakeNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"CorefakeNet loaded (epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc={checkpoint.get('best_val_acc', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("CorefakeNet loaded (raw state_dict)")

    model.to(device).eval()
    return model


def load_ensemble(device):
    """Load all ensemble models."""
    from core_models.efficientnet_texture import EfficientNetTexture
    from core_models.frequency_cnn import FrequencyCNN, fft_to_tensor
    from core_models.fusion_mlp import FusionMLP
    from core_models.dinov2_auth_model import DINOv2AuthModel
    from core_models.efficientnet_auth_model import EfficientNetAuthModel
    from core_models.face_deepfake_model import FaceDeepfakeModel

    models = {}

    # Texture model (EfficientNet-B4)
    path = os.path.join(MODELS_DIR, "efficient.pth")
    if os.path.exists(path):
        m = EfficientNetTexture()
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.to(device).eval()
        models['texture'] = m
        print("  Loaded: EfficientNet-B4 Texture")

    # Frequency CNN
    path = os.path.join(MODELS_DIR, "frequency.pth")
    if os.path.exists(path):
        m = FrequencyCNN()
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.to(device).eval()
        models['frequency'] = m
        print("  Loaded: Frequency CNN")

    # DINOv2
    path = os.path.join(MODELS_DIR, "dinov2_auth_model.pth")
    if os.path.exists(path):
        m = DINOv2AuthModel()
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.to(device).eval()
        models['dino'] = m
        print("  Loaded: DINOv2")

    # EfficientNetV2-S
    path = os.path.join(MODELS_DIR, "efficientnet_auth_model.pth")
    if os.path.exists(path):
        m = EfficientNetAuthModel()
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.to(device).eval()
        models['efficientnet'] = m
        print("  Loaded: EfficientNetV2-S Auth")

    # Face deepfake (ResNet50)
    path = os.path.join(MODELS_DIR, "image_face_model.pth")
    if os.path.exists(path):
        m = FaceDeepfakeModel()
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.to(device).eval()
        models['face'] = m
        print("  Loaded: Face Deepfake (ResNet50)")

    # Fusion MLP
    path = os.path.join(MODELS_DIR, "fusion_mlp.pth")
    if os.path.exists(path):
        m = FusionMLP()
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.to(device).eval()
        models['fusion'] = m
        print("  Loaded: Fusion MLP")

    # ViT (HuggingFace)
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        vit = ViTForImageClassification.from_pretrained(model_id).to(device)
        vit.eval()
        proc = ViTImageProcessor.from_pretrained(model_id)
        models['vit'] = vit
        models['vit_processor'] = proc
        print("  Loaded: ViT Deepfake Detector")
    except Exception as e:
        print(f"  WARNING: ViT not loaded: {e}")

    return models


# ================================================================
#  Evaluation Functions
# ================================================================

def evaluate_corefakenet(model, eval_data, device):
    """
    Evaluate CorefakeNet on evaluation data.

    Returns:
        (predictions, confidences, head_scores_dict, labels)
    """
    from core_models.corefakenet import CorefakeNet

    preprocess = CorefakeNet.PREPROCESS
    predictions = []
    confidences = []
    labels = []
    head_scores_all = {name: [] for name in CorefakeNet.HEAD_NAMES}

    for img, label in tqdm(eval_data, desc="CorefakeNet eval"):
        tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor, return_all=True)

        score = out['final_score'].item()
        conf = out['confidence'].item()
        predictions.append(score)
        confidences.append(conf)
        labels.append(label)

        for i, name in enumerate(CorefakeNet.HEAD_NAMES):
            head_scores_all[name].append(out['head_scores'][0, i].item())

    return (
        np.array(predictions),
        np.array(confidences),
        {k: np.array(v) for k, v in head_scores_all.items()},
        np.array(labels),
    )


def evaluate_ensemble(models, eval_data, device):
    """
    Evaluate the multi-model ensemble on evaluation data.

    Returns:
        (predictions, per_model_scores, labels)
    """
    from core_models.frequency_cnn import fft_to_tensor

    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    predictions = []
    per_model = {k: [] for k in ['vit', 'texture', 'frequency', 'dino',
                                  'efficientnet', 'face']}
    labels = []

    for img, label in tqdm(eval_data, desc="Ensemble eval"):
        img_rgb = img.convert("RGB")
        tensor = transform_224(img_rgb).unsqueeze(0).to(device)

        scores = {}

        with torch.no_grad():
            # ViT
            if 'vit' in models and 'vit_processor' in models:
                try:
                    inputs = models['vit_processor'](
                        images=img_rgb, return_tensors="pt"
                    ).to(device)
                    logits = models['vit'](**inputs).logits
                    probs = torch.softmax(logits, dim=1)
                    fake_idx = [
                        k for k, v in models['vit'].config.id2label.items()
                        if "fake" in v.lower()
                    ]
                    scores['vit'] = (
                        probs[0][fake_idx[0]].item()
                        if fake_idx else probs[0][1].item()
                    )
                except Exception:
                    scores['vit'] = 0.5

            # Texture
            if 'texture' in models:
                scores['texture'] = models['texture'](tensor).item()

            # Frequency
            if 'frequency' in models:
                fft_t = fft_to_tensor(img_rgb).unsqueeze(0).to(device)
                scores['frequency'] = models['frequency'](fft_t).item()

            # DINOv2
            if 'dino' in models:
                scores['dino'] = models['dino'](tensor).item()

            # EfficientNetV2
            if 'efficientnet' in models:
                scores['efficientnet'] = models['efficientnet'](tensor).item()

            # Face
            if 'face' in models:
                real_prob = models['face'](tensor).item()
                scores['face'] = 1.0 - real_prob

        for k in per_model:
            per_model[k].append(scores.get(k, 0.5))

        # Use FusionMLP if available, else weighted average
        if 'fusion' in models:
            final = models['fusion'].predict(
                vit=scores.get('vit', 0.0),
                efficientnet=scores.get('texture', 0.0),
                forensic=0.0,
                frequency=scores.get('frequency', 0.0),
            )
        else:
            w = {'vit': 0.40, 'texture': 0.20, 'frequency': 0.10,
                 'dino': 0.05, 'efficientnet': 0.15, 'face': 0.10}
            total_w = sum(w[k] for k in scores if k in w)
            final = sum(
                w.get(k, 0) * v for k, v in scores.items()
            ) / max(total_w, 1e-8)

        predictions.append(final)
        labels.append(label)

    return (
        np.array(predictions),
        {k: np.array(v) for k, v in per_model.items()},
        np.array(labels),
    )


def compute_metrics(predictions, labels, threshold=0.5):
    """Compute classification metrics."""
    pred_binary = (predictions > threshold).astype(int)
    labels_int = labels.astype(int)

    metrics = {
        'accuracy': accuracy_score(labels_int, pred_binary),
        'precision': precision_score(labels_int, pred_binary, zero_division=0),
        'recall': recall_score(labels_int, pred_binary, zero_division=0),
        'f1': f1_score(labels_int, pred_binary, zero_division=0),
    }

    try:
        metrics['roc_auc'] = roc_auc_score(labels_int, predictions)
    except ValueError:
        metrics['roc_auc'] = 0.0

    metrics['confusion_matrix'] = confusion_matrix(labels_int, pred_binary)
    return metrics


def benchmark_speed(model, device, model_type='corefakenet'):
    """
    Benchmark inference speed.

    Returns:
        dict with per_image_ms and estimated video times
    """
    from core_models.corefakenet import CorefakeNet

    if model_type == 'corefakenet':
        dummy = torch.randn(1, 3, 380, 380).to(device)
    else:
        dummy = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(dummy) if model_type != 'ensemble' else None

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(SPEED_ITERATIONS):
            start = time.perf_counter()
            model(dummy)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)

    # Video estimates (22s video)
    # At 6 fps sampling = 132 frames, at 0.5 fps = 11 frames
    return {
        'per_image_ms': round(avg_ms, 1),
        'per_image_std_ms': round(std_ms, 1),
        'video_22s_6fps_sec': round(avg_ms * 132 / 1000, 1),
        'video_22s_05fps_sec': round(avg_ms * 11 / 1000, 1),
    }


def benchmark_ensemble_speed(models, device):
    """Benchmark full ensemble inference speed."""
    from core_models.frequency_cnn import fft_to_tensor
    from PIL import Image

    dummy_224 = torch.randn(1, 3, 224, 224).to(device)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            for name, m in models.items():
                if name in ('vit_processor', 'fusion'):
                    continue
                if name == 'vit':
                    inputs = models['vit_processor'](
                        images=dummy_img, return_tensors="pt"
                    ).to(device)
                    m(**inputs)
                elif name == 'frequency':
                    fft_t = fft_to_tensor(dummy_img).unsqueeze(0).to(device)
                    m(fft_t)
                else:
                    m(dummy_224)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(SPEED_ITERATIONS):
            start = time.perf_counter()

            for name, m in models.items():
                if name in ('vit_processor', 'fusion'):
                    continue
                if name == 'vit':
                    inputs = models['vit_processor'](
                        images=dummy_img, return_tensors="pt"
                    ).to(device)
                    m(**inputs)
                elif name == 'frequency':
                    fft_t = fft_to_tensor(dummy_img).unsqueeze(0).to(device)
                    m(fft_t)
                else:
                    m(dummy_224)

            if 'fusion' in models:
                models['fusion'].predict(vit=0.5, efficientnet=0.5,
                                         forensic=0.0, frequency=0.5)

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)

    return {
        'per_image_ms': round(avg_ms, 1),
        'per_image_std_ms': round(std_ms, 1),
        'video_22s_6fps_sec': round(avg_ms * 132 / 1000, 1),
        'video_22s_05fps_sec': round(avg_ms * 11 / 1000, 1),
    }


# ================================================================
#  Main
# ================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 70)
    print("CorefakeNet vs Ensemble Evaluation")
    print("=" * 70)

    # ---- Load evaluation data ----
    # Use a held-out set (different seed + skip to avoid training data overlap)
    print("\nLoading evaluation dataset...")
    eval_data, _ = load_portrait_dataset(
        max_samples=EVAL_SAMPLES,
        train_split=1.0,
        face_align=True,
        skip_per_class=500,
        seed=999,
    )
    print(f"Evaluation samples: {len(eval_data)}")

    fake_count = sum(1 for _, l in eval_data if l == 1)
    real_count = sum(1 for _, l in eval_data if l == 0)
    print(f"  Fake: {fake_count}, Real: {real_count}")

    # ---- Load models ----
    print("\n--- Loading CorefakeNet ---")
    cfn_model = load_corefakenet(device)

    print("\n--- Loading Ensemble Models ---")
    ensemble_models = load_ensemble(device)

    # ---- Evaluate CorefakeNet ----
    if cfn_model is not None:
        print("\n" + "=" * 70)
        print("COREFAKENET EVALUATION")
        print("=" * 70)

        cfn_preds, cfn_conf, cfn_heads, cfn_labels = evaluate_corefakenet(
            cfn_model, eval_data, device,
        )

        cfn_metrics = compute_metrics(cfn_preds, cfn_labels)

        print(f"\n  Accuracy:  {cfn_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:   {cfn_metrics['roc_auc']:.4f}")
        print(f"  Precision: {cfn_metrics['precision']:.4f}")
        print(f"  Recall:    {cfn_metrics['recall']:.4f}")
        print(f"  F1:        {cfn_metrics['f1']:.4f}")

        cm = cfn_metrics['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted Real  Predicted Fake")
        print(f"    Actual Real    {cm[0][0]:>10d}    {cm[0][1]:>10d}")
        print(f"    Actual Fake    {cm[1][0]:>10d}    {cm[1][1]:>10d}")

        print(f"\n  Per-head mean scores (fake samples):")
        fake_mask = cfn_labels == 1
        real_mask = cfn_labels == 0
        for name in ['texture', 'frequency', 'artifact', 'vit', 'dino']:
            fake_mean = cfn_heads[name][fake_mask].mean() if fake_mask.any() else 0
            real_mean = cfn_heads[name][real_mask].mean() if real_mask.any() else 0
            print(f"    {name:12s}: fake={fake_mean:.4f}, real={real_mean:.4f}, "
                  f"gap={fake_mean - real_mean:+.4f}")

        print(f"\n  Mean confidence: {cfn_conf.mean():.4f}")

        # Speed benchmark
        print("\n  Speed benchmark...")
        cfn_speed = benchmark_speed(cfn_model, device, 'corefakenet')
        print(f"    Per image:           {cfn_speed['per_image_ms']:.1f} ms "
              f"(+/- {cfn_speed['per_image_std_ms']:.1f})")
        print(f"    Video 22s @ 6fps:    {cfn_speed['video_22s_6fps_sec']:.1f} sec")
        print(f"    Video 22s @ 0.5fps:  {cfn_speed['video_22s_05fps_sec']:.1f} sec")

    # ---- Evaluate Ensemble ----
    if ensemble_models:
        print("\n" + "=" * 70)
        print("ENSEMBLE EVALUATION")
        print("=" * 70)

        ens_preds, ens_per_model, ens_labels = evaluate_ensemble(
            ensemble_models, eval_data, device,
        )

        ens_metrics = compute_metrics(ens_preds, ens_labels)

        print(f"\n  Accuracy:  {ens_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:   {ens_metrics['roc_auc']:.4f}")
        print(f"  Precision: {ens_metrics['precision']:.4f}")
        print(f"  Recall:    {ens_metrics['recall']:.4f}")
        print(f"  F1:        {ens_metrics['f1']:.4f}")

        cm = ens_metrics['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted Real  Predicted Fake")
        print(f"    Actual Real    {cm[0][0]:>10d}    {cm[0][1]:>10d}")
        print(f"    Actual Fake    {cm[1][0]:>10d}    {cm[1][1]:>10d}")

        print(f"\n  Per-model mean scores (fake samples):")
        fake_mask = ens_labels == 1
        real_mask = ens_labels == 0
        for name in ens_per_model:
            scores = ens_per_model[name]
            fake_mean = scores[fake_mask].mean() if fake_mask.any() else 0
            real_mean = scores[real_mask].mean() if real_mask.any() else 0
            print(f"    {name:12s}: fake={fake_mean:.4f}, real={real_mean:.4f}, "
                  f"gap={fake_mean - real_mean:+.4f}")

        # Speed benchmark
        print("\n  Speed benchmark...")
        ens_speed = benchmark_ensemble_speed(ensemble_models, device)
        print(f"    Per image:           {ens_speed['per_image_ms']:.1f} ms "
              f"(+/- {ens_speed['per_image_std_ms']:.1f})")
        print(f"    Video 22s @ 6fps:    {ens_speed['video_22s_6fps_sec']:.1f} sec")
        print(f"    Video 22s @ 0.5fps:  {ens_speed['video_22s_05fps_sec']:.1f} sec")

    # ---- Comparison ----
    if cfn_model is not None and ensemble_models:
        print("\n" + "=" * 70)
        print("COMPARISON: CorefakeNet vs Ensemble")
        print("=" * 70)

        print(f"\n  {'Metric':<15s} {'CorefakeNet':>12s} {'Ensemble':>12s} {'Delta':>12s}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        for metric in ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']:
            cfn_val = cfn_metrics[metric]
            ens_val = ens_metrics[metric]
            delta = cfn_val - ens_val
            marker = " <--" if abs(delta) > 0.05 else ""
            print(f"  {metric:<15s} {cfn_val:>12.4f} {ens_val:>12.4f} "
                  f"{delta:>+12.4f}{marker}")

        print(f"\n  {'Speed':<15s} {'CorefakeNet':>12s} {'Ensemble':>12s} {'Speedup':>12s}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        speedup = ens_speed['per_image_ms'] / max(cfn_speed['per_image_ms'], 0.1)
        print(f"  {'per image':<15s} {cfn_speed['per_image_ms']:>10.1f}ms "
              f"{ens_speed['per_image_ms']:>10.1f}ms "
              f"{speedup:>11.1f}x")
        print(f"  {'video 6fps':<15s} {cfn_speed['video_22s_6fps_sec']:>10.1f}s  "
              f"{ens_speed['video_22s_6fps_sec']:>10.1f}s  "
              f"{ens_speed['video_22s_6fps_sec'] / max(cfn_speed['video_22s_6fps_sec'], 0.1):>11.1f}x")
        print(f"  {'video 0.5fps':<15s} {cfn_speed['video_22s_05fps_sec']:>10.1f}s  "
              f"{ens_speed['video_22s_05fps_sec']:>10.1f}s  "
              f"{ens_speed['video_22s_05fps_sec'] / max(cfn_speed['video_22s_05fps_sec'], 0.1):>11.1f}x")

        # Verdict
        acc_delta = cfn_metrics['accuracy'] - ens_metrics['accuracy']
        print(f"\n  VERDICT:")
        if acc_delta >= -0.05:
            print(f"    CorefakeNet accuracy is within 5% of ensemble ({acc_delta:+.4f})")
            print(f"    --> PASS: Ready for integration as Fast Mode")
        else:
            print(f"    CorefakeNet accuracy is {abs(acc_delta):.4f} below ensemble")
            print(f"    --> NEEDS IMPROVEMENT before integration")

    print(f"\n{'='*70}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
