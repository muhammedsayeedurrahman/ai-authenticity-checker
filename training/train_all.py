"""
Master training pipeline for ProofyX deepfake detection.

Trains all models in the correct order:
  1. DINOv2 auth model               -> models/dinov2_auth_model.pth
  2. EfficientNetV2-S auth model     -> models/efficientnet_auth_model.pth
  3. Face deepfake (ResNet50)        -> models/image_face_model.pth
  4. EfficientNet-B4 Texture branch  -> models/efficient.pth
  5. Frequency CNN                   -> models/frequency.pth
  6. Audio Deepfake CNN              -> models/audio_deepfake_model.pth
  7. Fusion MLP (needs 4 & 5 done)   -> models/fusion_mlp.pth
  8. CorefakeNet (unified hybrid)    -> models/corefakenet.pth

Note: ViT is loaded pre-trained from HuggingFace (no local training needed).
      Forensic analysis is heuristic (no training needed).

Usage:
    python training/train_all.py
"""

import subprocess
import sys
import os
import time
import glob

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_script(name, script_path):
    """Run a training script and return success status."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"Script:   {script_path}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-u", script_path],
        cwd=ROOT_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n{name} completed in {elapsed:.0f}s")
    else:
        print(f"\nWARNING: {name} exited with code {result.returncode} ({elapsed:.0f}s)")

    return result.returncode == 0


def main():
    print("ProofyX Training Pipeline")
    print("=" * 60)

    os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)

    steps = [
        ("DINOv2 Auth", "training/train_dinov2.py"),
        ("EfficientNetV2-S Auth", "training/train_efficientnet_auth.py"),
        ("Face Deepfake (ResNet50)", "training/train_face_deepfake_hf.py"),
        ("EfficientNet-B4 Texture", "training/train_efficientnet_texture.py"),
        ("Frequency CNN", "training/train_frequency_cnn.py"),
        ("Audio Deepfake CNN", "training/train_audio_deepfake.py"),
        ("Fusion MLP", "training/train_fusion.py"),
        ("CorefakeNet (Unified Hybrid)", "training/train_corefakenet.py"),
    ]

    results = {}
    for name, script in steps:
        success = run_script(name, script)
        results[name] = success
        if not success:
            print(f"WARNING: {name} failed")
            if name == "Fusion MLP":
                print("  Fusion depends on component models — check earlier steps")

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")

    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name:40s} [{status}]")

    print(f"\nModel files:")
    for f in sorted(glob.glob(os.path.join(ROOT_DIR, "models", "*.pth"))):
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {os.path.basename(f):40s} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
