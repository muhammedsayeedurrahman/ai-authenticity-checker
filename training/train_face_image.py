import sys
import os
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from core_models.face_deepfake_model import FaceDeepfakeModel

# ---------------- CONFIG ----------------
DATA_DIR = "data/image"     # expects data/image/real , data/image/fake
BATCH_SIZE = 32
EPOCHS = 15
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
MODEL_PATH = "models/image_face_model.pth"
EARLY_STOPPING_PATIENCE = 3
TRAIN_SPLIT = 0.9
# ---------------------------------------


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

    # -------- Dataset --------
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=None)

    train_size = int(len(full_dataset) * TRAIN_SPLIT)
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, tfm):
            self.subset = subset
            self.tfm = tfm

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            img = self.tfm(img)
            return img, label

    train_loader = DataLoader(
        TransformDataset(train_subset, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        TransformDataset(val_subset, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Train samples: {train_size}")
    print(f"Val samples  : {val_size}")

    # -------- Model --------
    model = FaceDeepfakeModel().to(device)
    criterion = nn.BCELoss()

    # Differential learning rates: backbone (layer4) vs classifier head
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # -------- Training with early stopping --------
    print("\nTraining Face Deepfake Model...\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            # ImageFolder: 0=fake, 1=real -> model outputs P(REAL)
            labels = labels.float().unsqueeze(1).to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
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
                labels_t = labels.float().unsqueeze(1).to(device)

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
