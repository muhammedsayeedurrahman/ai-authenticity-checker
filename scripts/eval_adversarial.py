"""
Adversarial robustness evaluation for ProofyX.

Tests model accuracy under common real-world degradations:
  - JPEG compression (quality 30-85)
  - Resize degradation (simulates screenshots)
  - Gaussian blur (removes frequency artifacts)
  - Social media compression (resize + JPEG combined)

Measures accuracy drop per perturbation type to identify model weaknesses.

Usage:
    python scripts/eval_adversarial.py
    python scripts/eval_adversarial.py --samples 1000
"""

import sys
import os
import json
import argparse
import random
from datetime import datetime, timezone
from copy import deepcopy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT_DIR, ".hf_cache", "datasets"))

import torch
from PIL import Image, ImageFilter

from training.evaluate import evaluate_models, load_portrait_dataset
from training.dataset_portraits import _jpeg_compress


# ──────────────────────────────────────────────
# Perturbation Functions
# ──────────────────────────────────────────────

def perturb_jpeg(img: Image.Image, quality: int = 50) -> Image.Image:
    """Apply JPEG compression at given quality."""
    return _jpeg_compress(img, quality)


def perturb_resize(img: Image.Image, scale: float = 0.5) -> Image.Image:
    """Downscale and upscale to simulate screenshot degradation."""
    w, h = img.size
    small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def perturb_blur(img: Image.Image, radius: float = 1.5) -> Image.Image:
    """Apply Gaussian blur to remove frequency artifacts."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def perturb_social_media(img: Image.Image) -> Image.Image:
    """Simulate social media compression: resize + JPEG."""
    w, h = img.size
    small = img.resize((int(w * 0.6), int(h * 0.6)), Image.BILINEAR)
    resized = small.resize((w, h), Image.BILINEAR)
    return _jpeg_compress(resized, quality=65)


PERTURBATIONS = {
    "clean": lambda img: img,
    "jpeg_q30": lambda img: perturb_jpeg(img, quality=30),
    "jpeg_q50": lambda img: perturb_jpeg(img, quality=50),
    "jpeg_q85": lambda img: perturb_jpeg(img, quality=85),
    "resize_0.3x": lambda img: perturb_resize(img, scale=0.3),
    "resize_0.5x": lambda img: perturb_resize(img, scale=0.5),
    "blur_r1.0": lambda img: perturb_blur(img, radius=1.0),
    "blur_r2.0": lambda img: perturb_blur(img, radius=2.0),
    "social_media": perturb_social_media,
}


def apply_perturbation(samples, perturbation_fn):
    """Apply a perturbation function to all images in a sample list."""
    perturbed = []
    for img, label in samples:
        try:
            new_img = perturbation_fn(img.convert("RGB"))
            perturbed.append((new_img, label))
        except Exception:
            perturbed.append((img, label))
    return perturbed


def main():
    parser = argparse.ArgumentParser(
        description="ProofyX Adversarial Robustness Evaluation",
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of evaluation samples (default: 500)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load evaluation dataset
    print(f"\nLoading evaluation dataset ({args.samples} samples)...")
    eval_data, _ = load_portrait_dataset(
        max_samples=args.samples,
        train_split=1.0,
        face_align=False,
        skip_per_class=5000,
        seed=999,
    )
    print(f"Evaluation set: {len(eval_data)} samples")

    all_results = {}

    for pert_name, pert_fn in PERTURBATIONS.items():
        print(f"\n{'=' * 60}")
        print(f"  Perturbation: {pert_name}")
        print(f"{'=' * 60}")

        perturbed_data = apply_perturbation(eval_data, pert_fn)
        results = evaluate_models(perturbed_data, device)

        if results:
            all_results[pert_name] = results

    # Compute accuracy drops relative to clean
    if "clean" in all_results:
        print(f"\n{'=' * 80}")
        print("  ACCURACY DROP ANALYSIS (relative to clean)")
        print(f"{'=' * 80}")

        clean = all_results["clean"]
        model_names = sorted(clean.keys())

        header = f"{'Perturbation':<18s}"
        for name in model_names:
            short = name[:12]
            header += f" {short:>12s}"
        print(header)
        print("-" * len(header))

        for pert_name, pert_results in all_results.items():
            if pert_name == "clean":
                continue
            row = f"{pert_name:<18s}"
            for name in model_names:
                if name in pert_results and name in clean:
                    drop = pert_results[name]["accuracy"] - clean[name]["accuracy"]
                    row += f" {drop:>+11.4f}"
                else:
                    row += f" {'N/A':>12s}"
            print(row)

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = os.path.join(ROOT_DIR, args.output)
    else:
        results_dir = os.path.join(ROOT_DIR, "evaluation", "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"adversarial_{timestamp}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "samples": len(eval_data),
        "perturbations": list(PERTURBATIONS.keys()),
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
