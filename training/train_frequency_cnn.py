"""
Train Frequency CNN on FFT magnitude spectra.

Fixes the preprocessing scaling mismatch that caused near-zero scores
in the old heuristic FrequencyAnalyzer.

Key fix: log(1 + abs(fft)) + per-image normalization to [0, 1]

Dataset: Multi-source portrait dataset (same as other models)
Output:  models/frequency.pth

Usage:
    python training/train_frequency_cnn.py
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

from core_models.frequency_cnn import FrequencyCNN
from training.dataset_portraits import (
    load_portrait_dataset, PortraitDataset,
)

# -------- CONFIG --------
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
MAX_SAMPLES = 3000
TRAIN_SPLIT = 0.85
MODEL_PATH = "models/frequency.pth"
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.05
# -------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------- Dataset (FFT mode) --------
    print("\nLoading portrait dataset for frequency training...")
    train_data, val_data = load_portrait_dataset(
        max_samples=MAX_SAMPLES,
        train_split=TRAIN_SPLIT,
        face_align=True,
    )

    # fft_mode=True converts images to FFT magnitude tensors
    # fft_augment=True applies random spatial transforms before FFT to create
    # training diversity (each epoch sees different FFT spectra)
    train_loader = DataLoader(
        PortraitDataset(train_data, fft_mode=True, fft_augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        PortraitDataset(val_data, fft_mode=True, fft_augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # -------- Model --------
    model = FrequencyCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # -------- Training --------
    print(f"\nTraining Frequency CNN (label_smoothing={LABEL_SMOOTHING})...\n")

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for fft_tensors, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            fft_tensors = fft_tensors.to(device)
            labels = labels.unsqueeze(1).to(device)
            smoothed = labels * (1 - LABEL_SMOOTHING) + (1 - labels) * LABEL_SMOOTHING

            preds = model(fft_tensors)
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
            for fft_tensors, labels in val_loader:
                fft_tensors = fft_tensors.to(device)
                labels_t = labels.unsqueeze(1).to(device)
                preds = model(fft_tensors)
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

    print(f"\nFrequency CNN trained. Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
