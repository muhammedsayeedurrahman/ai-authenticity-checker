"""
Dataset loader for AI-generated portrait detection.

Streams balanced datasets from HuggingFace:
  - GAN-generated faces (StyleGAN, etc.)
  - Diffusion-generated images (Stable Diffusion, Midjourney-style)
  - Real portrait photos from diverse sources

All images are optionally face-aligned and cropped before being returned.

Usage:
    from training.dataset_portraits import load_portrait_dataset

    train_data, val_data = load_portrait_dataset(
        max_samples=40000,
        train_split=0.85,
    )
"""

import sys
import os
import random

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT_DIR, ".hf_cache", "datasets"))

from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
from datasets import load_dataset


# ──────────────────────────────────────────────
# HuggingFace Dataset Sources
# ──────────────────────────────────────────────
# Each entry: dataset_id, split, image_col, label_col, fake_value, real_value
# Ordered by quality/diversity. Loader tries each source and skips failures.

PORTRAIT_SOURCES = [
    # --- Primary: large, well-labeled datasets ---
    {
        "id": "JamieWithofs/Deepfake-and-real-images",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,   # 0=Fake, 1=Real
        "real_value": 1,
        "description": "GAN + diffusion deepfakes vs real faces",
    },
    {
        "id": "Hemg/deepfake-and-real-images",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,
        "real_value": 1,
        "description": "190K deepfake and real images",
    },
    {
        "id": "Hemg/AI-Generated-vs-Real-Images-Datasets",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,   # 1=AI, 0=Real
        "real_value": 0,
        "description": "AI-generated vs real (diverse AI methods)",
    },
    # --- Secondary: modern AI generators, diffusion models ---
    {
        "id": "poloclub/diffusiondb",
        "name": "random_1k",
        "split": "train",
        "image_col": "image",
        "label_col": None,   # All images are AI-generated (no label col)
        "fake_value": None,
        "real_value": None,
        "all_fake": True,     # Every image is AI-generated
        "description": "Stable Diffusion generated images (DiffusionDB)",
    },
    {
        "id": "AIML-TUDA/i_RAVEN",
        "split": "test",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,
        "real_value": 0,
        "description": "AI vs real visual patterns",
    },
    {
        "id": "clips/deepfake_detection",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,
        "real_value": 0,
        "description": "Deepfake detection benchmark",
    },
    # --- Modern generators: diffusion, DALL-E, Midjourney ---
    {
        "id": "Rajarshi-Roy-research/Defactify_Image_Dataset",
        "split": "train",
        "image_col": "Image",
        "label_col": "Label_A",
        "fake_value": 1,   # 1=AI-generated, 0=Real
        "real_value": 0,
        "description": "96K images: SD2.1, SDXL, SD3, DALL-E 3, Midjourney v6",
    },
    {
        "id": "ComplexDataLab/OpenFake",
        "name": "core",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": "fake",   # string labels
        "real_value": "real",
        "description": "2.3M real vs AI-generated (multi-generator, multi-source)",
    },
    {
        "id": "DataScienceProject/Art_Images_Ai_And_Real_",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,   # 0=fake, 1=real
        "real_value": 1,
        "description": "AI art vs real images (balanced)",
    },
    {
        "id": "ehristoforu/midjourney-images",
        "split": "train",
        "image_col": "image",
        "label_col": None,
        "fake_value": None,
        "real_value": None,
        "all_fake": True,
        "description": "Midjourney V5/V6 generated images",
    },
    # --- Tertiary: additional real-face sources for balance ---
    {
        "id": "nielsr/CelebA-faces",
        "split": "train",
        "image_col": "image",
        "label_col": None,
        "fake_value": None,
        "real_value": None,
        "all_real": True,     # Every image is real
        "description": "CelebA real celebrity faces",
    },
    {
        "id": "logasja/UTKFace",
        "split": "train",
        "image_col": "image",
        "label_col": None,
        "fake_value": None,
        "real_value": None,
        "all_real": True,
        "description": "UTKFace real faces (diverse demographics)",
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
    except Exception:  # Broad catch: face detection can fail in many ways (cv2, PIL, etc.)
        pass
    return pil_img


def collect_from_source(source, per_class, face_align=True, skip_per_class=0):
    """
    Collect balanced samples from a single HuggingFace dataset source.

    Handles three source types:
      - Standard labeled (fake_value/real_value columns)
      - all_fake=True (every image is AI-generated, label=1)
      - all_real=True (every image is real, label=0)

    Returns list of (PIL.Image, label) where label is 0=real, 1=fake (normalized).
    """
    dataset_id = source["id"]
    print(f"  Loading {dataset_id}...")
    try:
        load_kwargs = {
            "path": dataset_id,
            "split": source["split"],
            "streaming": True,
        }
        if "name" in source:
            load_kwargs["name"] = source["name"]
        stream = load_dataset(**load_kwargs).shuffle(seed=42, buffer_size=10000)
    except Exception as e:  # Broad catch: HF dataset loading can fail in many ways
        print(f"  WARNING: Could not load {dataset_id}: {e}")
        return []

    is_all_fake = source.get("all_fake", False)
    is_all_real = source.get("all_real", False)
    label_col = source.get("label_col")

    fake_samples = []
    real_samples = []
    fake_skipped = 0
    real_skipped = 0
    total_seen = 0

    for sample in stream:
        # Determine label
        if is_all_fake:
            normalized_label = 1
        elif is_all_real:
            normalized_label = 0
        elif label_col is not None:
            raw_label = sample[label_col]
            fake_val = source["fake_value"]
            real_val = source["real_value"]

            # Support both string labels ("real"/"fake") and int labels (0/1)
            if isinstance(fake_val, str):
                raw_str = str(raw_label).lower().strip()
                if raw_str == fake_val.lower().strip():
                    normalized_label = 1
                elif raw_str == real_val.lower().strip():
                    normalized_label = 0
                else:
                    total_seen += 1
                    continue
            else:
                raw_int = int(raw_label)
                if raw_int == fake_val:
                    normalized_label = 1
                elif raw_int == real_val:
                    normalized_label = 0
                else:
                    total_seen += 1
                    continue
        else:
            total_seen += 1
            continue

        # Get image
        img_col = source.get("image_col", "image")
        try:
            img = sample[img_col].convert("RGB")
        except Exception:  # Broad catch: image decoding varies widely across sources
            total_seen += 1
            continue

        # Minimum size filter — skip tiny images
        if img.width < 64 or img.height < 64:
            total_seen += 1
            continue

        if normalized_label == 1:
            if fake_skipped < skip_per_class:
                fake_skipped += 1
            elif len(fake_samples) < per_class:
                if face_align:
                    img = _try_detect_face_crop(img)
                fake_samples.append((img, 1))
        else:
            if real_skipped < skip_per_class:
                real_skipped += 1
            elif len(real_samples) < per_class:
                if face_align:
                    img = _try_detect_face_crop(img)
                real_samples.append((img, 0))

        total_seen += 1

        if len(fake_samples) >= per_class and len(real_samples) >= per_class:
            break
        # For all_fake/all_real sources, only one side fills
        if is_all_fake and len(fake_samples) >= per_class:
            break
        if is_all_real and len(real_samples) >= per_class:
            break

        # Safety cap: don't scan indefinitely
        if total_seen > (per_class + skip_per_class) * 30:
            break

    print(f"    Collected: {len(fake_samples)} fake, {len(real_samples)} real "
          f"(scanned {total_seen}, skipped {fake_skipped}+{real_skipped})")
    return fake_samples + real_samples


def load_portrait_dataset(max_samples=40000, train_split=0.85, face_align=True,
                          skip_per_class=0, seed=42, sources=None):
    """
    Load balanced portrait dataset from multiple sources.

    Args:
        max_samples: Total target samples (split across sources).
                     Default increased to 40K for meaningful training.
        train_split: Fraction for training
        face_align: Apply face detection and cropping
        skip_per_class: Skip this many samples per class per source before
                        collecting (prevents overlap with other training sets)
        seed: Random seed for shuffling and splitting
        sources: Optional list of source dicts to use. Defaults to PORTRAIT_SOURCES.

    Returns:
        (train_samples, val_samples) -- each is list of (PIL.Image, label)
        label: 0=real, 1=fake (AI-generated)
    """
    if sources is None:
        sources = PORTRAIT_SOURCES

    # Filter to labeled sources (have both fake and real, or are marked all_fake/all_real)
    labeled_sources = [s for s in sources if
                       s.get("label_col") is not None or
                       s.get("all_fake") or
                       s.get("all_real")]

    per_source = max_samples // max(len(labeled_sources), 1)
    per_class = per_source // 2

    print(f"Loading portrait dataset: {max_samples} target samples from "
          f"{len(labeled_sources)} sources ({per_class} per class per source)"
          + (f", skipping {skip_per_class}/class/source" if skip_per_class else ""))

    all_samples = []
    for source in labeled_sources:
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
        raise RuntimeError(
            f"No balanced samples collected. Fake: {len(fake)}, Real: {len(real)}. "
            "Check dataset availability and network connection."
        )

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


# ──────────────────────────────────────────────
# Adversarial Augmentation
# ──────────────────────────────────────────────


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Simulate JPEG compression at a given quality level."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class AdversarialAugmentation:
    """
    Augmentations that simulate real-world degradations adversarial to
    deepfake detectors: JPEG compression, resize degradation, blur,
    and social media compression pipelines.

    Applied with probability p (default 0.4) during training.
    """

    def __init__(self, p: float = 0.4):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        aug_type = random.choice(["jpeg", "resize", "blur", "social_media"])

        if aug_type == "jpeg":
            quality = random.randint(30, 85)
            return _jpeg_compress(img, quality)

        elif aug_type == "resize":
            w, h = img.size
            scale = random.uniform(0.3, 0.7)
            small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)

        elif aug_type == "blur":
            radius = random.uniform(0.5, 2.5)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        else:  # social_media: resize + JPEG combined
            w, h = img.size
            scale = random.uniform(0.5, 0.8)
            small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            resized = small.resize((w, h), Image.BILINEAR)
            quality = random.randint(50, 80)
            return _jpeg_compress(resized, quality)


# ──────────────────────────────────────────────
# Standard transforms
# ──────────────────────────────────────────────

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
    train_data, val_data = load_portrait_dataset(max_samples=200, face_align=False)
    print(f"\nQuick test: {len(train_data)} train, {len(val_data)} val")
    img, label = train_data[0]
    print(f"Sample: {img.size}, label={label}")
