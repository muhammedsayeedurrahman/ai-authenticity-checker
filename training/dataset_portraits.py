"""
Dataset loader for AI-generated portrait detection.

Streams balanced datasets from HuggingFace:
  - Diffusion-generated faces
  - GAN portraits (StyleGAN, etc.)
  - Real portrait photos

All images are face-aligned and cropped before being returned.

Usage:
    from training.dataset_portraits import load_portrait_dataset

    train_data, val_data = load_portrait_dataset(
        max_samples=4000,
        train_split=0.85,
    )
"""

import sys
import os
import io
import random

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT_DIR, ".hf_cache", "datasets"))

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
from datasets import load_dataset


# -------- HuggingFace dataset sources --------
# Each entry: (dataset_id, split, image_col, label_col, fake_label_value)
PORTRAIT_SOURCES = [
    # Deepfake vs real faces (GAN + diffusion)
    {
        "id": "JamieWithofs/Deepfake-and-real-images",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,  # 0=Fake, 1=Real
        "real_value": 1,
    },
    # AI-Generated vs Real (diverse AI methods)
    {
        "id": "Hemg/AI-Generated-vs-Real-Images-Datasets",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,  # 1=AI, 0=Real
        "real_value": 0,
    },
    # 190k deepfake and real images
    {
        "id": "Hemg/deepfake-and-real-images",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,  # 0=Fake, 1=Real
        "real_value": 1,
    },
]


def _try_detect_face_crop(pil_img, expand_ratio=0.3):
    """
    Attempt face detection and cropping. Returns original if no face found.
    Uses OpenCV DNN face detector.
    """
    try:
        from utils.gradcam import detect_and_align_face
        face_crop, bbox = detect_and_align_face(pil_img, expand_ratio=expand_ratio)
        if face_crop is not None:
            return face_crop
    except Exception:
        pass
    return pil_img


def collect_from_source(source, per_class, face_align=True, skip_per_class=0):
    """
    Collect balanced samples from a single HuggingFace dataset source.

    Args:
        source: dict with dataset config
        per_class: target samples per class
        face_align: whether to apply face detection and cropping
        skip_per_class: skip this many samples per class before collecting
                        (used to avoid overlap with component model training data)

    Returns:
        list of (PIL.Image, label) where label is 0=real, 1=fake (normalized)
    """
    print(f"  Loading {source['id']}...")
    try:
        stream = load_dataset(
            source["id"],
            split=source["split"],
            streaming=True,
        ).shuffle(seed=42, buffer_size=10000)
    except Exception as e:
        print(f"  WARNING: Could not load {source['id']}: {e}")
        return []

    fake_samples = []
    real_samples = []
    fake_skipped = 0
    real_skipped = 0
    total_seen = 0

    for sample in stream:
        raw_label = int(sample[source["label_col"]])

        if raw_label == source["fake_value"]:
            if fake_skipped < skip_per_class:
                fake_skipped += 1
            elif len(fake_samples) < per_class:
                img = sample[source["image_col"]].convert("RGB")
                if face_align:
                    img = _try_detect_face_crop(img)
                fake_samples.append((img, 1))  # Normalized: 1=fake
        elif raw_label == source["real_value"]:
            if real_skipped < skip_per_class:
                real_skipped += 1
            elif len(real_samples) < per_class:
                img = sample[source["image_col"]].convert("RGB")
                if face_align:
                    img = _try_detect_face_crop(img)
                real_samples.append((img, 0))  # Normalized: 0=real

        total_seen += 1

        if len(fake_samples) >= per_class and len(real_samples) >= per_class:
            break

        if total_seen > (per_class + skip_per_class) * 20:
            break

    print(f"    Collected: {len(fake_samples)} fake, {len(real_samples)} real "
          f"(scanned {total_seen}, skipped {fake_skipped}+{real_skipped})")
    return fake_samples + real_samples


def load_portrait_dataset(max_samples=4000, train_split=0.85, face_align=True,
                          skip_per_class=0, seed=42):
    """
    Load balanced portrait dataset from multiple sources.

    Args:
        max_samples: Total target samples (split across sources)
        train_split: Fraction for training
        face_align: Apply face detection and cropping
        skip_per_class: Skip this many samples per class per source before
                        collecting (prevents overlap with other training sets)
        seed: Random seed for shuffling and splitting

    Returns:
        (train_samples, val_samples) — each is list of (PIL.Image, label)
        label: 0=real, 1=fake (AI-generated)
    """
    per_source = max_samples // len(PORTRAIT_SOURCES)
    per_class = per_source // 2

    print(f"Loading portrait dataset: {max_samples} target samples from "
          f"{len(PORTRAIT_SOURCES)} sources ({per_class} per class per source)"
          + (f", skipping {skip_per_class}/class/source" if skip_per_class else ""))

    all_samples = []
    for source in PORTRAIT_SOURCES:
        samples = collect_from_source(
            source, per_class, face_align=face_align,
            skip_per_class=skip_per_class,
        )
        all_samples.extend(samples)

    # Balance classes
    fake = [s for s in all_samples if s[1] == 1]
    real = [s for s in all_samples if s[1] == 0]
    min_count = min(len(fake), len(real))
    if min_count == 0:
        raise RuntimeError("No samples collected from any source")

    random.seed(seed)
    random.shuffle(fake)
    random.shuffle(real)
    balanced = fake[:min_count] + real[:min_count]
    random.shuffle(balanced)

    print(f"Balanced dataset: {len(balanced)} samples "
          f"({min_count} fake + {min_count} real)")

    # Split
    split_idx = int(len(balanced) * train_split)
    train_samples = balanced[:split_idx]
    val_samples = balanced[split_idx:]

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")
    return train_samples, val_samples


class PortraitDataset(Dataset):
    """
    PyTorch Dataset wrapper for portrait samples.

    Args:
        data: list of (PIL.Image, label)
        transform: torchvision transform to apply
        fft_mode: if True, return FFT magnitude instead of RGB tensor
        fft_augment: if True (and fft_mode=True), apply random spatial
                     transforms before FFT to create training diversity
    """

    def __init__(self, data, transform=None, fft_mode=False, fft_augment=False):
        self.data = data
        self.transform = transform
        self.fft_mode = fft_mode
        self.fft_augment = fft_augment and fft_mode

        if self.fft_augment:
            self.pre_fft_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop(256, scale=(0.85, 1.0)),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ], p=0.2),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        if self.fft_mode:
            from core_models.frequency_cnn import fft_to_tensor
            if self.fft_augment:
                img = self.pre_fft_transform(img)
            tensor = fft_to_tensor(img, size=256)
        elif self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        return tensor, torch.tensor(label, dtype=torch.float32)


# -------- Standard transforms --------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


if __name__ == "__main__":
    train_data, val_data = load_portrait_dataset(max_samples=100, face_align=False)
    print(f"\nQuick test: {len(train_data)} train, {len(val_data)} val")
    img, label = train_data[0]
    print(f"Sample: {img.size}, label={label}")
