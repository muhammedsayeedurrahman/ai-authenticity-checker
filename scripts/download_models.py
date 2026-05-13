#!/usr/bin/env python3
"""
Download HuggingFace models and document local .pth file requirements.

Reads configs/models.json to determine which models to download.
HuggingFace models are cached to .hf_cache/.
Local .pth files must be placed manually in the models/ directory.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --cache-dir /path/to/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    config_path = ROOT_DIR / "configs" / "models.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


def download_huggingface_models(config: dict, cache_dir: Path) -> None:
    """Download all HuggingFace models defined in the config."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    hf_models = []
    for key, value in config.items():
        if not isinstance(value, dict):
            continue
        if value.get("type") == "huggingface" and value.get("enabled", True):
            hf_models.append((key, value["model_id"]))

    if not hf_models:
        print("No HuggingFace models found in config.")
        return

    print(f"\nDownloading {len(hf_models)} HuggingFace model(s) to {cache_dir}/\n")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name, model_id in hf_models:
        print(f"  [{name}] Downloading {model_id}...")
        try:
            path = snapshot_download(
                model_id,
                cache_dir=str(cache_dir),
            )
            print(f"  [{name}] Cached at: {path}")
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")

    print()


def list_local_models(config: dict) -> None:
    """Print required local .pth files and their status."""
    models_dir = ROOT_DIR / "models"
    local_models = []

    for key, value in config.items():
        if not isinstance(value, dict):
            continue
        if value.get("type") == "local" and value.get("enabled", True):
            path = value.get("path", "")
            local_models.append((key, path, value.get("description", "")))

    if not local_models:
        print("No local .pth models found in config.")
        return

    print(f"Local model files (expected in {models_dir}/):\n")
    print(f"  {'Name':<20} {'File':<30} {'Status':<10} Description")
    print(f"  {'─' * 20} {'─' * 30} {'─' * 10} {'─' * 30}")

    for name, filename, desc in local_models:
        full_path = models_dir / filename
        status = "FOUND" if full_path.exists() else "MISSING"
        status_color = status
        print(f"  {name:<20} {filename:<30} {status_color:<10} {desc}")

    missing = [m for m in local_models if not (models_dir / m[1]).exists()]
    if missing:
        print(f"\n  {len(missing)} model(s) missing. Place .pth files in {models_dir}/")
    else:
        print(f"\n  All {len(local_models)} local model(s) found.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ProofyX ML models")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT_DIR / ".hf_cache",
        help="HuggingFace cache directory (default: .hf_cache/)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only list model status without downloading",
    )
    args = parser.parse_args()

    config = load_config()

    print("=" * 60)
    print("ProofyX Model Manager")
    print("=" * 60)

    list_local_models(config)

    if not args.skip_download:
        download_huggingface_models(config, args.cache_dir)
    else:
        print("Skipping HuggingFace downloads (--skip-download)")

    print("Done.")


if __name__ == "__main__":
    main()
