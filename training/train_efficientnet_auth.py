import sys
import os
import random
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Redirect HF cache to D: drive (C: has no space)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache", "datasets")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from datasets import load_dataset

from core_models.efficientnet_auth_model import EfficientNetAuthModel

# ---------------- CONFIG ----------------
BATCH_SIZE = 32
EPOCHS = 15
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
TRAIN_SPLIT = 0.85
MAX_SAMPLES = 4000           # Faster training
MODEL_PATH = "models/efficientnet_auth_model.pth"
EARLY_STOPPING_PATIENCE = 3
LABEL_SMOOTHING = 0.05
# ---------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- Transforms (video-like augmentation) --------
    train_transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------- Load dataset via streaming (balanced) --------
    print("Loading Hugging Face dataset (streaming)...")
    stream = load_dataset(
        "Hemg/AI-Generated-vs-Real-Images-Datasets",
        split="train",
        streaming=True
    )

    per_class = MAX_SAMPLES // 2
    print(f"Collecting {per_class} samples per class ({MAX_SAMPLES} total)...")
    class_buckets = {0: [], 1: []}  # 0=Real, 1=AI
    total_seen = 0
    last_printed = 0
    for sample in stream:
        label = int(sample["label"])  # 1 = AI, 0 = Real
        if label in class_buckets and len(class_buckets[label]) < per_class:
            img = sample["image"].convert("RGB")
            class_buckets[label].append((img, float(label)))
        total_seen += 1
        collected = len(class_buckets[0]) + len(class_buckets[1])
        if collected >= last_printed + 500:
            last_printed = collected
            print(f"  collected {collected}/{MAX_SAMPLES} (Real: {len(class_buckets[0])}, AI: {len(class_buckets[1])}, scanned: {total_seen})")
        if total_seen % 10000 == 0 and collected == last_printed:
            print(f"  scanning... {total_seen} samples seen (Real: {len(class_buckets[0])}, AI: {len(class_buckets[1])})")
        if len(class_buckets[0]) >= per_class and len(class_buckets[1]) >= per_class:
            break

    samples = class_buckets[0] + class_buckets[1]
    print(f"Collected {len(samples)} samples (Real: {len(class_buckets[0])}, AI: {len(class_buckets[1])}) from {total_seen} streamed")

    # Shuffle before split
    random.seed(42)
    random.shuffle(samples)

    # -------- Train / Val split --------
    split = int(len(samples) * TRAIN_SPLIT)
    train_data = samples[:split]
    val_data = samples[split:]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples  : {len(val_data)}")

    # -------- Dataset Wrapper --------
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, data, tfm):
            self.data = data
            self.tfm = tfm

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img, label = self.data[idx]
            img = self.tfm(img)
            return img, torch.tensor(label, dtype=torch.float32)

    train_loader = DataLoader(
        ImageDataset(train_data, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        ImageDataset(val_data, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -------- Model --------
    model = EfficientNetAuthModel().to(device)
    criterion = nn.BCELoss()

    # Differential learning rates
    backbone_params = [p for p in model.model.features.parameters() if p.requires_grad]
    head_params = list(model.model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": 1e-4},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": 1e-4},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # -------- Training --------
    print(f"\nStarting EfficientNetV2 training (label_smoothing={LABEL_SMOOTHING})...\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
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
                loss = criterion(preds, labels_t)
                val_loss += loss.item()

                pred_labels = (preds > 0.5).float()
                correct += (pred_labels == labels_t).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"| Train Loss: {avg_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        # -------- Early stopping --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    print("\nEfficientNetV2 authenticity model trained successfully.")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
