"""
Train EfficientNet-B4 texture branch on face-aligned portrait crops.

Captures texture artifacts (skin smoothness, hair detail, boundary artifacts)
that complement ViT's structural analysis.

Dataset: Multi-source portrait dataset (GAN + diffusion + real)
Output:  models/efficient.pth

Usage:
    python training/train_efficientnet_texture.py
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from core_models.efficientnet_texture import EfficientNetTexture
from training.dataset_portraits import (
    load_portrait_dataset, PortraitDataset,
    TRAIN_TRANSFORM, VAL_TRANSFORM,
)

# -------- CONFIG --------
BATCH_SIZE = 8              # B4 is large, keep batch small for CPU
EPOCHS = 15
BACKBONE_LR = 5e-6
HEAD_LR = 5e-4
MAX_SAMPLES = 3000
TRAIN_SPLIT = 0.85
MODEL_PATH = "models/efficient.pth"
EARLY_STOPPING_PATIENCE = 4
LABEL_SMOOTHING = 0.05
# -------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------- Dataset --------
    print("\nLoading portrait dataset for texture training...")
    train_data, val_data = load_portrait_dataset(
        max_samples=MAX_SAMPLES,
        train_split=TRAIN_SPLIT,
        face_align=True,
    )

    train_loader = DataLoader(
        PortraitDataset(train_data, TRAIN_TRANSFORM),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        PortraitDataset(val_data, VAL_TRANSFORM),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # -------- Model --------
    model = EfficientNetTexture().to(device)
    criterion = nn.BCELoss()

    # Differential LR: frozen early layers won't update anyway
    trainable_backbone = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": trainable_backbone, "lr": BACKBONE_LR, "weight_decay": 1e-4},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": 1e-4},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # -------- Training --------
    print(f"\nTraining EfficientNet-B4 Texture (label_smoothing={LABEL_SMOOTHING})...\n")

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)
            smoothed = labels * (1 - LABEL_SMOOTHING) + (1 - labels) * LABEL_SMOOTHING

            preds = model(imgs)
            loss = criterion(preds, smoothed)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels_t = labels.unsqueeze(1).to(device)
                preds = model(imgs)
                val_loss += criterion(preds, labels_t).item()
                correct += ((preds > 0.5).float() == labels_t).sum().item()
                total += labels.size(0)

        avg_val = val_loss / len(val_loader)
        val_acc = correct / total
        scheduler.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"| Train: {avg_loss:.4f} | Val: {avg_val:.4f} | Acc: {val_acc:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved (val_loss={avg_val:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nEfficientNet-B4 Texture trained. Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
