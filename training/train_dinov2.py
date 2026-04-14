import sys
import os
from tqdm import tqdm

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

from core_models.dinov2_auth_model import DINOv2AuthModel

# ================= CONFIG =================
BATCH_SIZE = 16
EPOCHS = 15
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
TRAIN_SPLIT = 0.85
MAX_SAMPLES = 2000
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.05
MODEL_PATH = "models/dinov2_auth_model.pth"
# ========================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- Transforms --------
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
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

    # -------- Load dataset via streaming (avoids full download) --------
    print("Loading Hugging Face dataset (streaming)...")
    stream = load_dataset(
        "Hemg/AI-Generated-vs-Real-Images-Datasets",
        split="train",
        streaming=True
    ).shuffle(seed=42, buffer_size=5000)

    print(f"Collecting {MAX_SAMPLES} samples...")
    samples = []
    for i, sample in enumerate(stream):
        if i >= MAX_SAMPLES:
            break
        img = sample["image"].convert("RGB")
        label = sample["label"]  # 0=real, 1=AI
        samples.append((img, label))
    print(f"Collected {len(samples)} samples")

    # -------- Train / Val split --------
    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples  : {len(val_samples)}")

    # -------- Dataset wrapper --------
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, data, tfm):
            self.data = data
            self.tfm = tfm

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img, label = self.data[idx]
            return self.tfm(img), torch.tensor(label, dtype=torch.float32)

    train_loader = DataLoader(
        ImageDataset(train_samples, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        ImageDataset(val_samples, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -------- Class weighting --------
    labels = [s[1] for s in train_samples]
    real_count = labels.count(0)
    ai_count = labels.count(1)

    print(f"Class balance -> Real: {real_count}, AI: {ai_count}")

    # -------- Model --------
    model = DINOv2AuthModel().to(device)
    criterion = nn.BCELoss()

    # Differential learning rates: backbone vs classifier head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": 1e-4},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": 1e-4},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # -------- Training with early stopping --------
    print(f"\nStarting DINOv2 auth model training (label_smoothing={LABEL_SMOOTHING})...\n")

    best_val_loss = float("inf")
    patience_counter = 0

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

    print("\nDINOv2 auth model training complete.")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
