"""
Train the Fusion MLP on validation-set predictions from all models.

This script:
  1. Loads all trained component models (ViT, EfficientNet, Forensic/Frequency CNN)
  2. Runs them on a held-out validation set to collect per-model scores
  3. Trains the FusionMLP + ModelCalibrator to optimally combine scores

The fusion layer learns:
  - Per-model temperature calibration (Task 5)
  - Optimal combination weights via MLP (Task 1)

Input:  [vit, efficientnet, forensic, frequency, face, dino, texture]
Output: models/fusion_mlp.pth

Usage:
    python training/train_fusion.py
"""

import sys
import os
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache", "datasets")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from core_models.fusion_mlp import FusionMLP
from core_models.efficientnet_texture import EfficientNetTexture
from core_models.frequency_cnn import FrequencyCNN, fft_to_tensor
from core_models.dinov2_auth_model import DINOv2AuthModel
from core_models.efficientnet_auth_model import EfficientNetAuthModel
from core_models.face_deepfake_model import FaceDeepfakeModel
from training.dataset_portraits import (
    load_portrait_dataset, PortraitDataset, VAL_TRANSFORM,
)

# -------- CONFIG --------
FUSION_EPOCHS = 50
FUSION_LR = 1e-3
FUSION_BATCH = 64
MAX_SAMPLES = 10000         # 5x increase for fusion training
MODEL_PATH = "models/fusion_mlp.pth"
MODELS_DIR = "models"
# -------------------------


def _load_vit(device):
    """Load ViT deepfake detector from HuggingFace."""
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        model = ViTForImageClassification.from_pretrained(model_id).to(device)
        processor = ViTImageProcessor.from_pretrained(model_id)
        model.eval()
        return model, processor
    except Exception as e:  # Broad catch: HF model loading can fail in many ways
        print(f"WARNING: Could not load ViT: {e}")
        return None, None


def _load_efficientnet(device):
    """Load trained EfficientNet-B4 texture model."""
    path = os.path.join(MODELS_DIR, "efficient.pth")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    model = EfficientNetTexture().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_frequency(device):
    """Load trained Frequency CNN."""
    path = os.path.join(MODELS_DIR, "frequency.pth")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    model = FrequencyCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_face(device):
    """Load trained Face Deepfake model."""
    path = os.path.join(MODELS_DIR, "image_face_model.pth")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    model = FaceDeepfakeModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_dino(device):
    """Load trained DINOv2 model."""
    path = os.path.join(MODELS_DIR, "dinov2_auth_model.pth")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    model = DINOv2AuthModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_efficientnet_auth(device):
    """Load trained EfficientNet Auth model."""
    path = os.path.join(MODELS_DIR, "efficientnet_auth_model.pth")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    model = EfficientNetAuthModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def collect_predictions(device):
    """
    Run all component models on a held-out dataset and collect
    (per-model scores, true label) pairs for fusion training.

    Returns:
        scores_tensor: (N, 7) — [vit, efficientnet, forensic, frequency, face, dino, texture]
        labels_tensor: (N,) — 0=real, 1=fake
    """
    print("\n--- Collecting model predictions for fusion training ---\n")

    # Load models
    vit_model, vit_processor = _load_vit(device)
    eff_model = _load_efficientnet(device)
    freq_model = _load_frequency(device)
    face_model = _load_face(device)
    dino_model = _load_dino(device)
    effauth_model = _load_efficientnet_auth(device)

    # Load dataset — skip samples used by component models to avoid data leakage.
    # Component models (EfficientNet texture, Frequency CNN) train on the first
    # ~500 samples/class/source (3000 total, 3 sources, 500/class/source).
    # We skip those and collect fresh samples for fusion training.
    val_data, _ = load_portrait_dataset(
        max_samples=MAX_SAMPLES,
        train_split=1.0,
        face_align=True,
        skip_per_class=3000,  # Skip component training data (larger with 10x samples)
        seed=123,            # Different seed from component training
    )

    val_transform = VAL_TRANSFORM

    all_scores = []
    all_labels = []

    print(f"Collecting predictions from {len(val_data)} samples...")

    for img, label in tqdm(val_data, desc="Scoring"):
        # [vit, efficientnet, forensic, frequency, face, dino, texture]
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        img_rgb = img.convert("RGB")
        tensor = val_transform(img_rgb).unsqueeze(0).to(device)

        # ViT score
        if vit_model is not None and vit_processor is not None:
            try:
                inputs = vit_processor(
                    images=img_rgb, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    logits = vit_model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)
                    fake_idx = [
                        k for k, v in vit_model.config.id2label.items()
                        if "fake" in v.lower()
                    ]
                    scores[0] = (
                        probs[0][fake_idx[0]].item()
                        if fake_idx else probs[0][1].item()
                    )
            except Exception:  # Broad catch: ViT inference varies across image inputs
                pass

        # EfficientNet Auth score
        if effauth_model is not None:
            try:
                with torch.no_grad():
                    scores[1] = effauth_model(tensor).item()
            except Exception:  # Broad catch: model can fail on edge-case images
                pass

        # Forensic score (heuristic — kept as input feature)
        try:
            from core.pipeline import forensic_score
            scores[2] = forensic_score(img)
        except Exception:  # Broad catch: forensic heuristic may fail on unusual images
            scores[2] = 0.0

        # Frequency CNN score
        if freq_model is not None:
            try:
                fft_tensor = fft_to_tensor(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    scores[3] = freq_model(fft_tensor).item()
            except Exception:  # Broad catch: FFT can fail on edge-case images
                pass

        # Face model score
        if face_model is not None:
            try:
                with torch.no_grad():
                    real_prob = face_model(tensor).item()
                    scores[4] = 1.0 - real_prob  # P(real) -> P(fake)
            except Exception:  # Broad catch: face model can fail on non-face images
                pass

        # DINOv2 score
        if dino_model is not None:
            try:
                with torch.no_grad():
                    scores[5] = dino_model(tensor).item()
            except Exception:  # Broad catch: DINO can fail on edge-case images
                pass

        # Texture (EfficientNet-B4) score
        if eff_model is not None:
            try:
                with torch.no_grad():
                    scores[6] = eff_model(tensor).item()
            except Exception:  # Broad catch: texture model can fail on edge-case images
                pass

        all_scores.append(scores)
        all_labels.append(float(label))

    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)

    print(f"\nCollected {len(all_scores)} prediction vectors")
    names = ["vit", "eff_auth", "forensic", "freq", "face", "dino", "texture"]
    means = [f"{names[i]}={scores_tensor[:, i].mean():.3f}" for i in range(7)]
    print(f"Score means: {', '.join(means)}")
    print(f"Labels: {(labels_tensor == 1).sum().item()} fake, "
          f"{(labels_tensor == 0).sum().item()} real")

    return scores_tensor, labels_tensor


def train_fusion(scores, labels, device):
    """
    Train FusionMLP on collected model predictions.

    Args:
        scores: (N, 4) per-model scores
        labels: (N,) ground truth (0=real, 1=fake)
    """
    print(f"\n--- Training Fusion MLP ---\n")

    # Train/val split
    n = len(scores)
    split = int(n * 0.8)
    indices = torch.randperm(n)

    train_scores = scores[indices[:split]]
    train_labels = labels[indices[:split]]
    val_scores = scores[indices[split:]]
    val_labels = labels[indices[split:]]

    train_ds = TensorDataset(train_scores, train_labels)
    val_ds = TensorDataset(val_scores, val_labels)

    train_loader = DataLoader(train_ds, batch_size=FUSION_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=FUSION_BATCH, shuffle=False)

    # Model
    model = FusionMLP(n_inputs=7).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FUSION_LR, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=FUSION_EPOCHS, eta_min=1e-5)

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(FUSION_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_scores, batch_labels in train_loader:
            batch_scores = batch_scores.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)

            preds = model(batch_scores)
            loss = criterion(preds, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_scores, batch_labels in val_loader:
                batch_scores = batch_scores.to(device)
                batch_labels = batch_labels.unsqueeze(1).to(device)
                preds = model(batch_scores)
                val_loss += criterion(preds, batch_labels).item()
                correct += ((preds > 0.5).float() == batch_labels).sum().item()
                total += batch_labels.size(0)

        avg_val = val_loss / max(len(val_loader), 1)
        val_acc = correct / max(total, 1)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{FUSION_EPOCHS}] "
                  f"| Train: {avg_loss:.4f} | Val: {avg_val:.4f} | Acc: {val_acc:.4f}")

            # Print learned temperatures
            temps = model.calibrator.temperatures.data.cpu().numpy()
            print(f"  Temperatures: vit={temps[0]:.3f}, eff={temps[1]:.3f}, "
                  f"forensic={temps[2]:.3f}, freq={temps[3]:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            if patience >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Print final fusion weights
    print(f"\nFusion MLP trained. Saved to: {MODEL_PATH}")
    print(f"Learned temperatures: {model.calibrator.temperatures.data.cpu().numpy()}")
    print(f"Fusion layer weights:\n"
          f"  Layer 1: {model.fusion[0].weight.data.cpu().numpy()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scores, labels = collect_predictions(device)
    train_fusion(scores, labels, device)


if __name__ == "__main__":
    main()
