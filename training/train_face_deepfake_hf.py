import sys
import os
import io
import random
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Redirect HF cache to project drive (C: may have limited space)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache", "datasets")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from datasets import load_dataset
from PIL import Image, ImageFilter
import numpy as np

from core_models.face_deepfake_model import FaceDeepfakeModel

# ---------------- CONFIG ----------------
BATCH_SIZE = 16
EPOCHS = 25
BACKBONE_LR = 5e-6         # Lower LR for subtler features
HEAD_LR = 5e-4
TRAIN_SPLIT = 0.85
MAX_SAMPLES = 12000         # More data for better generalization
MODEL_PATH = "models/image_face_model.pth"
EARLY_STOPPING_PATIENCE = 5  # More patience
LABEL_SMOOTHING = 0.05       # Prevent overconfidence
# ---------------------------------------


class JPEGCompression:
    """Simulate video compression artifacts (critical for video deepfake detection)."""
    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class GaussianNoise:
    """Add Gaussian noise to simulate video noise/artifacts."""
    def __init__(self, std_range=(0.01, 0.05)):
        self.std_range = std_range

    def __call__(self, tensor):
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0, 1)


class RandomDownscaleUpscale:
    """Simulate resolution changes common in deepfake videos."""
    def __init__(self, scale_range=(0.5, 0.9)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale = random.uniform(*self.scale_range)
        w, h = img.size
        small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- Transforms --------
    # Aggressive augmentation to simulate video artifacts
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomApply([JPEGCompression(quality_range=(30, 90))], p=0.4),
        transforms.RandomApply([RandomDownscaleUpscale(scale_range=(0.5, 0.85))], p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.RandomApply([GaussianNoise(std_range=(0.01, 0.04))], p=0.3),
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
    print("Loading HuggingFace dataset (streaming)...")
    stream = load_dataset(
        "JamieWithofs/Deepfake-and-real-images",
        split="train",
        streaming=True
    )

    per_class = MAX_SAMPLES // 2
    print(f"Collecting {per_class} samples per class ({MAX_SAMPLES} total)...")
    class_buckets = {0: [], 1: []}  # 0=Fake, 1=Real
    total_seen = 0
    last_printed = 0
    for sample in stream:
        label = int(sample["label"])  # 0=Fake, 1=Real
        if len(class_buckets[label]) < per_class:
            img = sample["image"].convert("RGB")
            class_buckets[label].append((img, float(label)))
        total_seen += 1
        collected = len(class_buckets[0]) + len(class_buckets[1])
        if collected >= last_printed + 500:
            last_printed = collected
            print(f"  collected {collected}/{MAX_SAMPLES} (Fake: {len(class_buckets[0])}, Real: {len(class_buckets[1])}, scanned: {total_seen})")
        if total_seen % 10000 == 0 and collected == last_printed:
            print(f"  scanning... {total_seen} samples seen (Fake: {len(class_buckets[0])}, Real: {len(class_buckets[1])})")
        if len(class_buckets[0]) >= per_class and len(class_buckets[1]) >= per_class:
            break

    samples = class_buckets[0] + class_buckets[1]
    print(f"Collected {len(samples)} samples (Fake: {len(class_buckets[0])}, Real: {len(class_buckets[1])}) from {total_seen} streamed")

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
        pin_memory=False
    )

    val_loader = DataLoader(
        ImageDataset(val_data, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # -------- Model --------
    model = FaceDeepfakeModel().to(device)

    # Use BCELoss with label smoothing manually
    criterion = nn.BCELoss()

    # Differential learning rates
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": 1e-4},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": 1e-4},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # -------- Training --------
    print(f"\nStarting Face Deepfake Model training (label_smoothing={LABEL_SMOOTHING})...\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            # Apply label smoothing: 0 -> 0.05, 1 -> 0.95
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

        # -------- Validation (no label smoothing) --------
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

    print("\nFace Deepfake Model trained successfully.")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
