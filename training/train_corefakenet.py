"""
Train CorefakeNet: unified hybrid CNN for ProofyX deepfake detection.

Multi-task training with 5 specialized heads + attention fusion.
Uses the same multi-source portrait dataset as the component models.

Training features:
  - Differential learning rates (backbone < heads < fusion)
  - CosineAnnealingWarmRestarts scheduler
  - Multi-task loss with KL diversity regularization
  - Early stopping on validation accuracy
  - Label smoothing (0.05)
  - Gradient clipping (max_norm=1.0)
  - 380x380 input (EfficientNet-B4 optimal)

Output: models/corefakenet.pth

Usage:
    python training/train_corefakenet.py
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms

from core_models.corefakenet import CorefakeNet, corefakenet_loss
from training.dataset_portraits import load_portrait_dataset, PortraitDataset

# ================= CONFIG =================
BATCH_SIZE = 4                  # B4 at 380x380 is memory-heavy on CPU
EPOCHS = 30
MAX_SAMPLES = 3000              # CPU-practical; balanced across 3 HF sources
TRAIN_SPLIT = 0.85
MODEL_PATH = "models/corefakenet.pth"
EARLY_STOPPING_PATIENCE = 10
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
# ==========================================

# 380x380 transforms for CorefakeNet
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((420, 420)),
    transforms.RandomResizedCrop(380, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1,
    ),
    transforms.RandomRotation(10),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    ], p=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def train_epoch(model, loader, optimizer, device):
    """Run one training epoch. Returns (avg_loss, loss_breakdown, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_accum = {}

    for imgs, labels in tqdm(loader, desc="Train"):
        imgs = imgs.to(device)
        targets = labels.unsqueeze(1).to(device)

        # Label smoothing
        smoothed = targets * (1 - LABEL_SMOOTHING) + \
                   (1 - targets) * LABEL_SMOOTHING

        outputs = model(imgs, return_all=True)

        # Replace targets with smoothed for loss computation
        loss, breakdown = corefakenet_loss(outputs, smoothed)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        pred = (outputs['final_score'] > 0.5).float()
        total_correct += (pred == targets).sum().item()
        total_samples += targets.size(0)

        for k, v in breakdown.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    accuracy = total_correct / total_samples
    avg_breakdown = {k: v / n_batches for k, v in loss_accum.items()}

    return avg_loss, avg_breakdown, accuracy


@torch.no_grad()
def validate(model, loader, device):
    """Run validation. Returns (avg_loss, accuracy, loss_breakdown)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_accum = {}

    for imgs, labels in loader:
        imgs = imgs.to(device)
        targets = labels.unsqueeze(1).to(device)

        outputs = model(imgs, return_all=True)
        loss, breakdown = corefakenet_loss(outputs, targets)

        total_loss += loss.item()
        pred = (outputs['final_score'] > 0.5).float()
        total_correct += (pred == targets).sum().item()
        total_samples += targets.size(0)

        for k, v in breakdown.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

    n_batches = max(len(loader), 1)
    avg_loss = total_loss / n_batches
    accuracy = total_correct / max(total_samples, 1)
    avg_breakdown = {k: v / n_batches for k, v in loss_accum.items()}

    return avg_loss, accuracy, avg_breakdown


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CorefakeNet Training")
    print(f"=" * 60)

    # ---- Dataset ----
    print(f"\nLoading portrait dataset ({MAX_SAMPLES} samples)...")
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

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = CorefakeNet().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {total_params:,} total, {trainable_params:,} trainable")

    # ---- Optimizer with differential LR ----
    param_groups = model.get_param_groups(weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_groups)

    print("Learning rates:")
    for i, pg in enumerate(param_groups):
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  Group {i}: lr={pg['lr']:.1e}, params={n_params:,}")

    # ---- Scheduler ----
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # ---- Training loop ----
    print(f"\nStarting training (epochs={EPOCHS}, patience={EARLY_STOPPING_PATIENCE})...\n")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_breakdown, train_acc = train_epoch(
            model, train_loader, optimizer, device,
        )

        # Validate
        val_loss, val_acc, val_breakdown = validate(model, val_loader, device)

        scheduler.step()

        # Log
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )
        print(
            f"  Losses: head={val_breakdown.get('head_loss', 0):.4f} "
            f"fusion={val_breakdown.get('fusion_loss', 0):.4f} "
            f"div={val_breakdown.get('diversity_loss', 0):.4f} "
            f"conf={val_breakdown.get('confidence_loss', 0):.4f} "
            f"| Temp={model.temperature.item():.3f}"
        )

        # Early stopping on validation accuracy
        if val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'temperature': model.temperature.item(),
                'epoch': epoch + 1,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'model_config': {
                    'backbone': 'efficientnet-b4',
                    'input_size': 380,
                    'heads': CorefakeNet.HEAD_NAMES,
                    'fusion': 'attention',
                    'hidden_dim': model.hidden_dim,
                },
                'training_info': {
                    'max_samples': MAX_SAMPLES,
                    'batch_size': BATCH_SIZE,
                    'label_smoothing': LABEL_SMOOTHING,
                    'weight_decay': WEIGHT_DECAY,
                },
            }, MODEL_PATH)
            print(f"  -> Best model saved (acc={val_acc:.4f}, loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} "
                    f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
                )
                break

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss:     {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

    # Print final attention weights distribution
    model.eval()
    print(f"\nFinal temperature: {model.temperature.item():.4f}")

    # Quick sanity check on a val batch
    with torch.no_grad():
        sample_imgs, sample_labels = next(iter(val_loader))
        sample_imgs = sample_imgs.to(device)
        out = model(sample_imgs, return_all=True)
        attn = out['attention_weights'].mean(dim=0)
        print("Mean attention weights:")
        for i, name in enumerate(CorefakeNet.HEAD_NAMES):
            print(f"  {name:12s}: {attn[i].item():.4f}")

        head_means = out['head_scores'].mean(dim=0)
        print("Mean head scores (on val batch):")
        for i, name in enumerate(CorefakeNet.HEAD_NAMES):
            print(f"  {name:12s}: {head_means[i].item():.4f}")

    print(f"\nModel size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
