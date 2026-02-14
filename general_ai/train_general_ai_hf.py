import sys
import os
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from datasets import load_dataset

from general_ai.clip_model import GeneralAIImageDetector
from pipeline.face_gate import face_present

# ================= CONFIG =================
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
TRAIN_SPLIT = 0.9
MAX_SAMPLES = 60000
CACHE_FILE = "data/general_ai_non_face_cache.pt"
TEMP_IMAGE = "temp_face_check.jpg"
EARLY_STOPPING_PATIENCE = 3
# ========================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- Transforms --------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # -------- Load HF dataset --------
    print("Loading Hugging Face dataset...")
    dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")["train"]
    dataset = dataset.shuffle(seed=42)

    if MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))

    # -------- FACE FILTERING (CACHED) --------
    os.makedirs("data", exist_ok=True)

    if os.path.exists(CACHE_FILE):
        print("Loading cached non-face dataset...")
        filtered_samples = torch.load(CACHE_FILE, weights_only=False)
    else:
        print("Filtering out face images (first run only)...")
        filtered_samples = []

        for sample in tqdm(dataset):
            image = sample["image"]
            label = sample["label"]  # 0 = real, 1 = AI

            image_rgb = image.convert("RGB")
            image_rgb.save(TEMP_IMAGE)

            if not face_present(TEMP_IMAGE):
                filtered_samples.append((image_rgb, label))

        if os.path.exists(TEMP_IMAGE):
            os.remove(TEMP_IMAGE)

        torch.save(filtered_samples, CACHE_FILE)
        print("Face-filter cache saved.")

    print(f"Remaining non-face samples: {len(filtered_samples)}")

    # -------- Train / Validation split --------
    split_idx = int(len(filtered_samples) * TRAIN_SPLIT)
    train_data = filtered_samples[:split_idx]
    val_data = filtered_samples[split_idx:]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples  : {len(val_data)}")

    # -------- Dataset wrapper --------
    class HFImageDataset(torch.utils.data.Dataset):
        def __init__(self, data, tfm):
            self.data = data
            self.tfm = tfm

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image, label = self.data[idx]
            image = self.tfm(image)
            return image, torch.tensor(label, dtype=torch.float32)

    train_loader = DataLoader(
        HFImageDataset(train_data, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        HFImageDataset(val_data, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # -------- Model --------
    model = GeneralAIImageDetector(device).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # -------- Training with early stopping --------
    print("\nStarting training...\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.unsqueeze(1).to(device)

                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item()

                predicted = (preds > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # -------- Early stopping --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/general_ai_model.pth")
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    print("\nGeneral AI image detector trained successfully.")
    print("Best model saved to: models/general_ai_model.pth")


if __name__ == "__main__":
    main()
