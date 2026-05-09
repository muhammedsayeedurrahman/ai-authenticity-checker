"""
ProofyX benchmark runner.

Wraps training/evaluate.py and outputs structured JSON results with
a markdown summary table.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --samples 5000
    python scripts/run_benchmarks.py --output evaluation/results/custom.json
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT_DIR, ".hf_cache", "datasets"))

import torch

from training.evaluate import evaluate_models, load_portrait_dataset


def generate_markdown_table(results: dict) -> str:
    """Generate a markdown table from evaluation results."""
    header = "| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |"
    separator = "|-------|----------|-----------|--------|------|---------|"
    rows = [header, separator]

    for name, m in sorted(results.items(), key=lambda x: -x[1].get("f1", 0)):
        row = (
            f"| {name} "
            f"| {m['accuracy']:.4f} "
            f"| {m['precision']:.4f} "
            f"| {m['recall']:.4f} "
            f"| {m['f1']:.4f} "
            f"| {m['auc_roc']:.4f} |"
        )
        rows.append(row)

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="ProofyX Benchmark Runner")
    parser.add_argument(
        "--samples", type=int, default=2000,
        help="Number of evaluation samples (default: 2000)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: evaluation/results/benchmark_<date>.json)",
    )
    parser.add_argument(
        "--skip-per-class", type=int, default=5000,
        help="Skip training samples to avoid data leakage (default: 5000)",
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
        skip_per_class=args.skip_per_class,
        seed=999,
    )
    print(f"Evaluation set: {len(eval_data)} samples")

    # Run evaluation
    results = evaluate_models(eval_data, device)

    if not results:
        print("No models available for evaluation.")
        return

    # Build output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "samples": len(eval_data),
        "models": results,
        "markdown_table": generate_markdown_table(results),
    }

    # Determine output path
    if args.output:
        output_path = os.path.join(ROOT_DIR, args.output)
    else:
        results_dir = os.path.join(ROOT_DIR, "evaluation", "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"benchmark_{timestamp}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"\n{output['markdown_table']}")


if __name__ == "__main__":
    main()
