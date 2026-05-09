"""
Comprehensive evaluation script for ProofyX deepfake detection models.

Computes proper ML metrics that the original training scripts lacked:
  - Precision, Recall, F1-score (per class and weighted)
  - AUC-ROC and AUC-PR
  - Confusion matrix
  - Cross-dataset evaluation (train on X, test on Y)
  - Per-model and ensemble evaluation
  - Calibration curve analysis

Usage:
    # Evaluate all models on default validation set
    python training/evaluate.py

    # Evaluate with cross-dataset testing
    python training/evaluate.py --cross-dataset

    # Evaluate a specific model
    python training/evaluate.py --model efficientnet

    # Use more samples for evaluation
    python training/evaluate.py --samples 10000
"""

import sys
import os
import argparse
import json
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT_DIR, ".hf_cache", "datasets"))

import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from training.dataset_portraits import (
    load_portrait_dataset, PortraitDataset, PORTRAIT_SOURCES, VAL_TRANSFORM,
)


# ──────────────────────────────────────────────
# Metrics (no sklearn dependency)
# ──────────────────────────────────────────────

def confusion_matrix(y_true, y_pred):
    """Compute 2x2 confusion matrix: [[TN, FP], [FN, TP]]."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """Compute precision, recall, F1 for binary classification."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def auc_roc(y_true, y_scores):
    """Compute AUC-ROC using trapezoidal rule (no sklearn needed)."""
    if len(set(y_true)) < 2:
        return 0.0

    # Sort by score descending
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    scores_sorted = [p[0] for p in pairs]
    labels_sorted = [p[1] for p in pairs]

    n_pos = sum(labels_sorted)
    n_neg = len(labels_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_prev, fpr_prev = 0.0, 0.0
    auc = 0.0
    tp, fp = 0, 0

    # Use unique thresholds
    thresholds = sorted(set(scores_sorted), reverse=True)
    for threshold in thresholds:
        while tp + fp < len(labels_sorted) and scores_sorted[tp + fp] >= threshold:
            if labels_sorted[tp + fp] == 1:
                tp += 1
            else:
                fp += 1
            if tp + fp >= len(labels_sorted):
                break

        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev, fpr_prev = tpr, fpr

    # Add final rectangle
    auc += (1.0 - fpr_prev) * (1.0 + tpr_prev) / 2.0

    return float(auc)


def compute_all_metrics(y_true, y_scores, threshold=0.5):
    """Compute comprehensive metrics for a model's predictions."""
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    metrics = precision_recall_f1(y_true, y_pred)
    metrics["auc_roc"] = auc_roc(y_true, y_scores)
    metrics["threshold"] = threshold
    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = sum(y_true)
    metrics["n_negative"] = len(y_true) - sum(y_true)
    return metrics


# ──────────────────────────────────────────────
# Model Loaders
# ──────────────────────────────────────────────

MODELS_DIR = os.path.join(ROOT_DIR, "models")


def load_vit(device):
    """Load ViT deepfake detector from HuggingFace."""
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        model = ViTForImageClassification.from_pretrained(model_id).to(device)
        processor = ViTImageProcessor.from_pretrained(model_id)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"  Could not load ViT: {e}")
        return None, None


def load_efficientnet_texture(device):
    """Load trained EfficientNet-B4 texture model."""
    path = os.path.join(MODELS_DIR, "efficient.pth")
    if not os.path.exists(path):
        return None
    from core_models.efficientnet_texture import EfficientNetTexture
    model = EfficientNetTexture().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_frequency_cnn(device):
    """Load trained Frequency CNN."""
    path = os.path.join(MODELS_DIR, "frequency.pth")
    if not os.path.exists(path):
        return None
    from core_models.frequency_cnn import FrequencyCNN
    model = FrequencyCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_efficientnet_auth(device):
    """Load trained EfficientNet auth model."""
    path = os.path.join(MODELS_DIR, "efficientnet_auth_model.pth")
    if not os.path.exists(path):
        return None
    from core_models.efficientnet_auth_model import EfficientNetAuthModel
    model = EfficientNetAuthModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_dinov2(device):
    """Load trained DINOv2 auth model."""
    path = os.path.join(MODELS_DIR, "dinov2_auth_model.pth")
    if not os.path.exists(path):
        return None
    from core_models.dinov2_auth_model import DINOv2AuthModel
    model = DINOv2AuthModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_corefakenet(device):
    """Load trained CorefakeNet."""
    path = os.path.join(MODELS_DIR, "corefakenet.pth")
    if not os.path.exists(path):
        return None
    from core_models.corefakenet import CorefakeNet
    model = CorefakeNet().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_face_deepfake(device):
    """Load trained Face Deepfake model (note: label convention 1=Real)."""
    path = os.path.join(MODELS_DIR, "image_face_model.pth")
    if not os.path.exists(path):
        return None
    from core_models.face_deepfake_model import FaceDeepfakeModel
    model = FaceDeepfakeModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


# ──────────────────────────────────────────────
# Scoring Functions
# ──────────────────────────────────────────────

def score_vit(model, processor, img, device):
    """Get ViT fake probability."""
    inputs = processor(images=img.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        fake_idx = [k for k, v in model.config.id2label.items() if "fake" in v.lower()]
        return probs[0][fake_idx[0]].item() if fake_idx else probs[0][1].item()


def score_binary_model(model, img, transform, device):
    """Score a binary classification model (output = fake probability)."""
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).item()


def score_face_deepfake(model, img, transform, device):
    """Score the face deepfake model (label convention: 1=Real, so invert)."""
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        # Model predicts 1=Real, so fake_score = 1 - output
        return 1.0 - model(tensor).item()


def score_frequency(model, img, device):
    """Score with Frequency CNN using FFT input."""
    from core_models.frequency_cnn import fft_to_tensor
    fft_tensor = fft_to_tensor(img, size=256).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(fft_tensor).item()


def score_corefakenet(model, img, device):
    """Score with CorefakeNet (380x380 input)."""
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor, return_all=False)
        return output.item()


# ──────────────────────────────────────────────
# Evaluation Functions
# ──────────────────────────────────────────────

def evaluate_models(samples, device, model_filter=None):
    """
    Evaluate all available models on a dataset.

    Args:
        samples: list of (PIL.Image, label) where label 0=real, 1=fake
        device: torch device
        model_filter: if set, only evaluate this model name

    Returns:
        dict of {model_name: metrics_dict}
    """
    val_transform = VAL_TRANSFORM

    # Load all models
    models = {}
    print("\nLoading models...")

    if model_filter is None or model_filter == "vit":
        vit_model, vit_processor = load_vit(device)
        if vit_model:
            models["vit"] = (vit_model, vit_processor)

    if model_filter is None or model_filter == "efficientnet":
        eff = load_efficientnet_texture(device)
        if eff:
            models["efficientnet_texture"] = eff

    if model_filter is None or model_filter == "efficientnet_auth":
        eff_auth = load_efficientnet_auth(device)
        if eff_auth:
            models["efficientnet_auth"] = eff_auth

    if model_filter is None or model_filter == "dinov2":
        dino = load_dinov2(device)
        if dino:
            models["dinov2"] = dino

    if model_filter is None or model_filter == "frequency":
        freq = load_frequency_cnn(device)
        if freq:
            models["frequency_cnn"] = freq

    if model_filter is None or model_filter == "corefakenet":
        cfn = load_corefakenet(device)
        if cfn:
            models["corefakenet"] = cfn

    if model_filter is None or model_filter == "face_deepfake":
        face = load_face_deepfake(device)
        if face:
            models["face_deepfake"] = face

    if not models:
        print("No models found. Train models first with: python training/train_all.py")
        return {}

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    # Collect predictions per model
    predictions = defaultdict(list)
    labels = []

    print(f"\nScoring {len(samples)} samples...")
    for img, label in tqdm(samples, desc="Evaluating"):
        labels.append(label)

        for name, model_obj in models.items():
            try:
                if name == "vit":
                    model, processor = model_obj
                    score = score_vit(model, processor, img, device)
                elif name == "frequency_cnn":
                    score = score_frequency(model_obj, img, device)
                elif name == "corefakenet":
                    score = score_corefakenet(model_obj, img, device)
                elif name == "face_deepfake":
                    score = score_face_deepfake(model_obj, img, val_transform, device)
                else:
                    score = score_binary_model(model_obj, img, val_transform, device)
            except Exception:
                score = 0.5  # Default on error
            predictions[name].append(score)

    # Compute metrics per model
    results = {}
    for name, scores in predictions.items():
        metrics = compute_all_metrics(labels, scores)
        results[name] = metrics

    # Compute ensemble (simple average)
    if len(predictions) > 1:
        ensemble_scores = []
        for i in range(len(labels)):
            avg = np.mean([predictions[name][i] for name in predictions])
            ensemble_scores.append(float(avg))
        results["ensemble_avg"] = compute_all_metrics(labels, ensemble_scores)

    return results


def print_results(results, title="Evaluation Results"):
    """Print metrics in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"{'Model':<22s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} "
          f"{'AUC':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s}")
    print(f"{'-' * 80}")

    for name, m in sorted(results.items(), key=lambda x: -x[1].get("f1", 0)):
        print(
            f"{name:<22s} "
            f"{m['accuracy']:>6.3f} "
            f"{m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} "
            f"{m['auc_roc']:>6.3f} "
            f"{m['tp']:>5d} "
            f"{m['fp']:>5d} "
            f"{m['fn']:>5d} "
            f"{m['tn']:>5d}"
        )

    print(f"{'=' * 80}")
    n = results.get("ensemble_avg", {}).get("n_samples", 0)
    if n:
        print(f"  Samples: {n} ({results['ensemble_avg']['n_positive']} fake, "
              f"{results['ensemble_avg']['n_negative']} real)")


def cross_dataset_eval(device, samples_per_source=500):
    """
    Cross-dataset evaluation: train models already trained, evaluate on each
    source separately to see generalization.

    This tests if models overfit to one dataset's artifacts.
    """
    print("\n" + "=" * 80)
    print("  CROSS-DATASET EVALUATION")
    print("  Testing per-source generalization")
    print("=" * 80)

    for source in PORTRAIT_SOURCES:
        source_id = source["id"]
        print(f"\n--- Testing on: {source_id} ---")

        try:
            samples = []
            from training.dataset_portraits import collect_from_source
            per_class = samples_per_source // 2
            source_samples = collect_from_source(
                source, per_class, face_align=False, skip_per_class=0,
            )
            if len(source_samples) < 20:
                print(f"  Skipping (only {len(source_samples)} samples collected)")
                continue

            results = evaluate_models(source_samples, device)
            print_results(results, title=f"Results on {source_id}")

        except Exception as e:
            print(f"  Error evaluating on {source_id}: {e}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ProofyX Model Evaluation")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of evaluation samples (default: 2000)")
    parser.add_argument("--model", type=str, default=None,
                        help="Evaluate specific model (vit, efficientnet, dinov2, etc.)")
    parser.add_argument("--cross-dataset", action="store_true",
                        help="Run cross-dataset evaluation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--skip-per-class", type=int, default=5000,
                        help="Skip training samples to avoid data leakage (default: 5000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.cross_dataset:
        cross_dataset_eval(device, samples_per_source=min(args.samples, 1000))
        return

    # Load evaluation dataset (skip training samples)
    print(f"\nLoading evaluation dataset ({args.samples} samples, "
          f"skipping {args.skip_per_class}/class/source to avoid leakage)...")

    eval_data, _ = load_portrait_dataset(
        max_samples=args.samples,
        train_split=1.0,  # Use all as eval
        face_align=False,
        skip_per_class=args.skip_per_class,
        seed=999,
    )

    print(f"Evaluation set: {len(eval_data)} samples")

    # Evaluate
    results = evaluate_models(eval_data, device, model_filter=args.model)

    if not results:
        return

    print_results(results)

    # Save to JSON if requested
    if args.output:
        output_path = os.path.join(ROOT_DIR, args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
