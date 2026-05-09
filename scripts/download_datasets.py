"""
Download and prepare all training datasets for ProofyX.

Pre-downloads HuggingFace datasets to local cache and optionally saves
sample images to local folders for easy browsing and demos.

Usage:
    # Download all datasets and save samples locally
    python scripts/download_datasets.py

    # Save more samples per dataset to local folders
    python scripts/download_datasets.py --samples 200

    # Verify datasets without downloading
    python scripts/download_datasets.py --verify-only

    # Show dataset statistics
    python scripts/download_datasets.py --stats

    # Skip local export (cache only)
    python scripts/download_datasets.py --no-local
"""

import sys
import os
import argparse
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")


def setup_cache(cache_dir=None):
    """Set up HuggingFace cache directories."""
    if cache_dir is None:
        cache_dir = os.path.join(ROOT_DIR, ".hf_cache")

    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "datasets"), exist_ok=True)
    return cache_dir


# ──────────────────────────────────────────────
# Image Datasets (synced with dataset_portraits.py)
# ──────────────────────────────────────────────

IMAGE_DATASETS = [
    {
        "id": "JamieWithofs/Deepfake-and-real-images",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,
        "real_value": 1,
        "description": "GAN + diffusion deepfakes vs real faces",
    },
    {
        "id": "Hemg/deepfake-and-real-images",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,
        "real_value": 1,
        "description": "190K deepfake and real images",
    },
    {
        "id": "Hemg/AI-Generated-vs-Real-Images-Datasets",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,
        "real_value": 0,
        "description": "AI-generated vs real (diverse AI methods)",
    },
    {
        "id": "poloclub/diffusiondb",
        "name": "random_1k",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": None,
        "all_fake": True,
        "description": "Stable Diffusion generated images (DiffusionDB)",
    },
    {
        "id": "AIML-TUDA/i_RAVEN",
        "split": "test",
        "type": "image",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,
        "real_value": 0,
        "description": "AI vs real visual patterns",
    },
    {
        "id": "clips/deepfake_detection",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,
        "real_value": 0,
        "description": "Deepfake detection benchmark",
    },
    {
        "id": "Rajarshi-Roy-research/Defactify_Image_Dataset",
        "split": "train",
        "type": "image",
        "image_col": "Image",
        "label_col": "Label_A",
        "fake_value": 1,
        "real_value": 0,
        "description": "96K images: SD2.1, SDXL, SD3, DALL-E 3, Midjourney v6",
    },
    {
        "id": "ComplexDataLab/OpenFake",
        "name": "core",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": "label",
        "fake_value": "fake",
        "real_value": "real",
        "description": "2.3M real vs AI-generated (multi-generator, multi-source)",
    },
    {
        "id": "DataScienceProject/Art_Images_Ai_And_Real_",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,
        "real_value": 1,
        "description": "AI art vs real images (balanced)",
    },
    {
        "id": "ehristoforu/midjourney-images",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": None,
        "all_fake": True,
        "description": "Midjourney V5/V6 generated images",
    },
    {
        "id": "nielsr/CelebA-faces",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": None,
        "all_real": True,
        "description": "CelebA real celebrity faces",
    },
    {
        "id": "logasja/UTKFace",
        "split": "train",
        "type": "image",
        "image_col": "image",
        "label_col": None,
        "all_real": True,
        "description": "UTKFace real faces (diverse demographics)",
    },
]

# ──────────────────────────────────────────────
# Audio Datasets
# ──────────────────────────────────────────────

AUDIO_DATASETS = [
    {
        "id": "moibrahimovic/fake_or_real",
        "split": "train",
        "type": "audio",
        "description": "Fake or real audio classification",
    },
    {
        "id": "ud-nlp/real-vs-fake-human-voice-deepfake-audio",
        "split": "train",
        "type": "audio",
        "description": "Real vs fake human voice deepfake audio",
    },
    {
        "id": "garystafford/deepfake-audio-detection",
        "split": "train",
        "type": "audio",
        "label_col": "label",
        "fake_value": 1,
        "real_value": 0,
        "description": "Deepfake audio detection (1,866 FLAC samples)",
    },
]

ALL_DATASETS = IMAGE_DATASETS + AUDIO_DATASETS


def verify_dataset(ds_info):
    """Verify a dataset is accessible by reading one sample."""
    from datasets import load_dataset

    ds_id = ds_info["id"]
    try:
        load_kwargs = {
            "path": ds_id,
            "split": ds_info["split"],
            "streaming": True,
        }
        if "name" in ds_info:
            load_kwargs["name"] = ds_info["name"]

        stream = load_dataset(**load_kwargs)
        sample = next(iter(stream))
        columns = list(sample.keys())
        return True, columns
    except Exception as e:
        return False, str(e)


def _classify_label(ds_info, sample):
    """Determine if a sample is 'real' or 'fake' based on dataset config.

    Returns 'real', 'fake', or None if unknown.
    """
    if ds_info.get("all_fake"):
        return "fake"
    if ds_info.get("all_real"):
        return "real"

    label_col = ds_info.get("label_col")
    if label_col is None or label_col not in sample:
        return None

    raw = sample[label_col]
    fake_val = ds_info.get("fake_value")
    real_val = ds_info.get("real_value")

    if isinstance(fake_val, str):
        raw_str = str(raw).lower().strip()
        if raw_str == fake_val.lower():
            return "fake"
        if raw_str == real_val.lower():
            return "real"
    else:
        try:
            raw_int = int(raw)
            if raw_int == fake_val:
                return "fake"
            if raw_int == real_val:
                return "real"
        except (ValueError, TypeError):
            pass

    return None


def download_and_export(ds_info, n_samples=50, save_local=True):
    """Download a dataset and optionally save sample images to local folders.

    Saves to:
      data/samples/<dataset_slug>/real/  — real images
      data/samples/<dataset_slug>/fake/  — fake/AI-generated images
    """
    from datasets import load_dataset

    ds_id = ds_info["id"]
    ds_type = ds_info["type"]

    try:
        load_kwargs = {
            "path": ds_id,
            "split": ds_info["split"],
            "streaming": True,
        }
        if "name" in ds_info:
            load_kwargs["name"] = ds_info["name"]

        stream = load_dataset(**load_kwargs)
    except Exception as e:
        return False, str(e), 0, 0

    # Local save directories
    slug = ds_id.replace("/", "_").replace("-", "_")
    real_dir = os.path.join(DATA_DIR, "samples", slug, "real")
    fake_dir = os.path.join(DATA_DIR, "samples", slug, "fake")

    if save_local and ds_type == "image":
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

    real_saved = 0
    fake_saved = 0
    per_class = n_samples // 2
    count = 0
    errors = 0

    image_col = ds_info.get("image_col", "image")

    for sample in stream:
        count += 1

        if ds_type == "image" and save_local:
            label = _classify_label(ds_info, sample)
            if label is None:
                if count > n_samples * 5:
                    break
                continue

            if label == "real" and real_saved >= per_class:
                if fake_saved >= per_class:
                    break
                continue
            if label == "fake" and fake_saved >= per_class:
                if real_saved >= per_class:
                    break
                continue

            try:
                img = sample[image_col]
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                else:
                    continue

                if label == "real":
                    save_path = os.path.join(real_dir, f"real_{real_saved:04d}.jpg")
                    real_saved += 1
                else:
                    save_path = os.path.join(fake_dir, f"fake_{fake_saved:04d}.jpg")
                    fake_saved += 1

                img.save(save_path, "JPEG", quality=90)
            except Exception:
                errors += 1
                if errors > 20:
                    break
                continue
        else:
            # Audio or cache-only: just iterate to warm cache
            if count >= n_samples:
                break

    return True, count, real_saved, fake_saved


def get_cache_stats(cache_dir):
    """Get cache directory statistics."""
    total_size = 0
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
                total_files += 1
            except OSError:
                pass
    return total_size, total_files


def get_local_stats():
    """Get local data/samples/ directory statistics."""
    samples_dir = os.path.join(DATA_DIR, "samples")
    if not os.path.isdir(samples_dir):
        return {}

    stats = {}
    for ds_slug in sorted(os.listdir(samples_dir)):
        ds_path = os.path.join(samples_dir, ds_slug)
        if not os.path.isdir(ds_path):
            continue

        real_dir = os.path.join(ds_path, "real")
        fake_dir = os.path.join(ds_path, "fake")

        real_count = len(os.listdir(real_dir)) if os.path.isdir(real_dir) else 0
        fake_count = len(os.listdir(fake_dir)) if os.path.isdir(fake_dir) else 0

        if real_count + fake_count > 0:
            stats[ds_slug] = {"real": real_count, "fake": fake_count}

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download ProofyX training datasets")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify dataset accessibility")
    parser.add_argument("--stats", action="store_true",
                        help="Show cache and local statistics")
    parser.add_argument("--samples", type=int, default=100,
                        help="Samples to save per dataset (split between real/fake)")
    parser.add_argument("--no-local", action="store_true",
                        help="Skip saving images to local folders")
    args = parser.parse_args()

    cache_dir = setup_cache(args.cache_dir)
    print(f"Cache directory: {cache_dir}")
    print(f"Local data dir:  {DATA_DIR}")

    if args.stats:
        size, files = get_cache_stats(cache_dir)
        print(f"\nCache statistics:")
        print(f"  Total size: {size / 1024 / 1024:.1f} MB")
        print(f"  Total files: {files}")

        local = get_local_stats()
        if local:
            print(f"\nLocal samples (data/samples/):")
            for slug, counts in local.items():
                print(f"  {slug:<50s} real={counts['real']:>4d}  fake={counts['fake']:>4d}")
        else:
            print(f"\nNo local samples yet. Run without --stats to download.")
        return

    save_local = not args.no_local

    print(f"\n{'=' * 70}")
    print(f"  ProofyX Dataset {'Verification' if args.verify_only else 'Download'}")
    print(f"  {len(ALL_DATASETS)} datasets ({len(IMAGE_DATASETS)} image, {len(AUDIO_DATASETS)} audio)")
    if save_local and not args.verify_only:
        print(f"  Saving {args.samples} sample images per dataset to data/samples/")
    print(f"{'=' * 70}\n")

    results = {}
    for ds_info in ALL_DATASETS:
        ds_id = ds_info["id"]
        ds_type = ds_info["type"]
        desc = ds_info["description"]

        print(f"[{ds_type:5s}] {ds_id}")
        print(f"        {desc}")

        start = time.time()

        if args.verify_only:
            ok, detail = verify_dataset(ds_info)
            elapsed = time.time() - start
            if ok:
                print(f"        OK ({elapsed:.1f}s) - Columns: {detail}")
            else:
                print(f"        FAILED ({elapsed:.1f}s) - {detail}")
            results[ds_id] = ok
        else:
            ok, count, real_n, fake_n = download_and_export(
                ds_info, n_samples=args.samples, save_local=save_local,
            )
            elapsed = time.time() - start
            if ok:
                detail = f"streamed {count}"
                if save_local and ds_type == "image":
                    detail += f", saved {real_n} real + {fake_n} fake locally"
                print(f"        OK ({elapsed:.1f}s) - {detail}")
            else:
                print(f"        FAILED ({elapsed:.1f}s) - {count}")
            results[ds_id] = ok

        print()

    # Summary
    ok_count = sum(1 for v in results.values() if v)
    fail_count = sum(1 for v in results.values() if not v)

    print(f"{'=' * 70}")
    print(f"  Summary: {ok_count} OK, {fail_count} FAILED out of {len(results)}")
    print(f"{'=' * 70}")

    if fail_count > 0:
        print("\n  Failed datasets:")
        for ds_id, ok in results.items():
            if not ok:
                print(f"    - {ds_id}")
        print("\n  These datasets will be skipped during training.")
        print("  The training pipeline handles missing sources gracefully.")

    if save_local and not args.verify_only:
        local = get_local_stats()
        if local:
            print(f"\n  Local samples saved to: {os.path.join(DATA_DIR, 'samples')}")
            print(f"  Browse the folders to see real vs fake examples:")
            for slug, counts in local.items():
                print(f"    {slug}/real/  ({counts['real']} images)")
                print(f"    {slug}/fake/  ({counts['fake']} images)")

    # Cache stats
    size, files = get_cache_stats(cache_dir)
    print(f"\n  Cache: {size / 1024 / 1024:.1f} MB, {files} files")


if __name__ == "__main__":
    main()
