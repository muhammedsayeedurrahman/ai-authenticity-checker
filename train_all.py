"""
Unified training script - trains all four models in sequence.
Run this to train Face, DINOv2, EfficientNet, and Audio models automatically.
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def main():
    print("=" * 60)
    print("DEEPFAKE DETECTION - UNIFIED TRAINING")
    print("=" * 60)
    print("\nThis will train all four models in sequence:")
    print("  1. Face Deepfake Model (~2-3 hours on GPU)")
    print("  2. DINOv2 Auth Model (~30-45 minutes on GPU)")
    print("  3. EfficientNet Auth Model (~20-30 minutes on GPU)")
    print("  4. Audio Deepfake CNN (~30-60 minutes on GPU)")
    print("\nTotal estimated time: 4-5 hours on GPU, 12-18 hours on CPU")
    print("=" * 60)

    response = input("\nProceed with training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return

    # Import training modules
    print("\n" + "=" * 60)
    print("STEP 1/4: Training Face Deepfake Model")
    print("=" * 60)
    from training.train_face_deepfake_hf import main as train_face
    try:
        train_face()
        print("\n✓ Face Deepfake Model training complete!")
    except Exception as e:
        print(f"\n✗ Face model training failed: {e}")
        print("Continuing with remaining models...")

    print("\n" + "=" * 60)
    print("STEP 2/4: Training DINOv2 Auth Model")
    print("=" * 60)
    from training.train_dinov2 import main as train_dino
    try:
        train_dino()
        print("\n✓ DINOv2 Auth Model training complete!")
    except Exception as e:
        print(f"\n✗ DINOv2 model training failed: {e}")
        print("Continuing with remaining models...")

    print("\n" + "=" * 60)
    print("STEP 3/4: Training EfficientNet Auth Model")
    print("=" * 60)
    from training.train_efficientnet_auth import main as train_eff
    try:
        train_eff()
        print("\n✓ EfficientNet Auth Model training complete!")
    except Exception as e:
        print(f"\n✗ EfficientNet model training failed: {e}")
        print("Continuing with remaining models...")

    print("\n" + "=" * 60)
    print("STEP 4/4: Training Audio Deepfake CNN")
    print("=" * 60)
    from training.train_audio_deepfake import main as train_audio
    try:
        train_audio()
        print("\n✓ Audio Deepfake CNN training complete!")
    except Exception as e:
        print(f"\n✗ Audio model training failed: {e}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nTrained models saved to: models/")
    print("\nTo run the application:")
    print("  python app.py")
    print("\nThen open: http://127.0.0.1:7861")
    print("=" * 60)

if __name__ == "__main__":
    main()
