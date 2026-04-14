"""
Training script for the Audio Deepfake CNN model.

Trains on the ASVspoof 2019 LA dataset (same as zo9999) via HuggingFace,
or falls back to a smaller public dataset if unavailable.

Architecture: 2-layer CNN on mel-spectrograms (matches zo9999 exactly).

Usage:
    python training/train_audio_deepfake.py
"""

import sys
import os
import random
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache", "datasets")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import librosa

from core_models.audio_deepfake_model import AudioDeepfakeCNN

# ================= CONFIG =================
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
TRAIN_SPLIT = 0.85
MAX_SAMPLES = 8000          # Total samples to collect
MODEL_PATH = "models/audio_deepfake_model.pth"
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.05

# Audio preprocessing (matches zo9999)
SAMPLE_RATE = 22050
N_MELS = 91
MAX_TIME_STEPS = 150
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 5.0          # seconds (matches zo9999 training)

# Dataset config — try multiple sources
HF_DATASETS = [
    "moibrahimovic/fake_or_real",
    "ud-nlp/real-vs-fake-human-voice-deepfake-audio",
]
# ==========================================


def audio_to_mel(waveform, sr=SAMPLE_RATE):
    """Convert waveform to mel-spectrogram (matching zo9999)."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate to fixed width
    if mel_db.shape[1] < MAX_TIME_STEPS:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])),
            mode="constant",
        )
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]

    return mel_db


class SpecAugment:
    """SpecAugment: frequency and time masking for mel-spectrogram augmentation."""

    def __init__(self, freq_mask_param=15, time_mask_param=20,
                 n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, mel):
        """Apply random frequency and time masks to a mel-spectrogram.

        Args:
            mel: numpy array of shape (n_mels, time_steps)
        Returns:
            augmented mel-spectrogram (same shape)
        """
        mel = mel.copy()
        n_mels, n_time = mel.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            mel[f0:f0 + f, :] = mel.mean()

        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_time - 1))
            t0 = random.randint(0, n_time - t)
            mel[:, t0:t0 + t] = mel.mean()

        return mel


class AudioMelDataset(Dataset):
    """Dataset that holds pre-computed mel-spectrograms and labels."""

    def __init__(self, mels, labels, augment=False):
        self.mels = mels
        self.labels = labels
        self.augment = augment
        if augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=12, time_mask_param=15,
                n_freq_masks=2, n_time_masks=2,
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        if self.augment:
            mel = self.spec_augment(mel)
            # Random gain variation
            if random.random() < 0.3:
                gain = random.uniform(0.8, 1.2)
                mel = mel * gain
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label


def _parse_label(sample):
    """Parse label from a HuggingFace dataset sample. Returns 0=fake, 1=real, or -1 if unknown."""
    # ASVspoof style: 'key' column
    if "key" in sample:
        raw = sample["key"]
        return 1 if raw == "bonafide" else 0

    # fake_or_real style: 'label' as string or int
    if "label" in sample:
        raw = sample["label"]
        if isinstance(raw, str):
            low = raw.lower().strip()
            if low in ("bonafide", "real", "genuine", "original", "authentic"):
                return 1
            elif low in ("spoof", "fake", "deepfake", "synthetic", "generated"):
                return 0
            return -1
        else:
            # int label: 0 = real/original, nonzero = fake (common convention)
            # But check if it's binary (0/1) where 1=fake
            return 0 if int(raw) > 0 else 1

    if "is_fake" in sample:
        return 0 if sample["is_fake"] else 1

    return -1


def load_dataset_hf():
    """Load audio samples from HuggingFace datasets (tries multiple sources)."""
    from datasets import load_dataset, Audio

    # Force soundfile backend (avoids torchcodec/FFmpeg issues on Windows)
    import datasets.config
    datasets.config.AUDIO_DECODER_BACKEND = "soundfile"

    ds = None
    for ds_name in HF_DATASETS:
        print(f"Attempting to load: {ds_name}")
        try:
            ds = load_dataset(ds_name, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=True))
            # Test that we can actually read a sample
            sample = next(iter(ds))
            print(f"  Columns: {list(sample.keys())}")
            # Re-create the iterator since we consumed one
            ds = load_dataset(ds_name, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=True))
            print(f"  Successfully connected to {ds_name}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            ds = None

    if ds is None:
        print("\nAll HuggingFace datasets failed.")
        print("Please provide audio data manually in data/audio/real/ and data/audio/fake/")
        return None, None

    per_class = MAX_SAMPLES // 2
    mels = []
    labels = []
    class_counts = {0: 0, 1: 0}  # 0=fake, 1=real
    total_seen = 0
    errors = 0

    print(f"Collecting {per_class} samples per class ({MAX_SAMPLES} total)...")

    for sample in ds:
        total_seen += 1

        label = _parse_label(sample)
        if label == -1:
            continue

        if class_counts[label] >= per_class:
            if all(c >= per_class for c in class_counts.values()):
                break
            continue

        try:
            audio_data = sample["audio"]
            waveform = audio_data["array"]
            sr = audio_data["sampling_rate"]

            if len(waveform) == 0:
                continue

            max_samples = int(MAX_DURATION * sr)
            waveform = waveform[:max_samples].astype(np.float32)

            mel = audio_to_mel(waveform, sr)
            mels.append(mel)
            labels.append(label)
            class_counts[label] += 1

            collected = class_counts[0] + class_counts[1]
            if collected % 200 == 0:
                print(f"  collected {collected}/{MAX_SAMPLES} "
                      f"(Fake: {class_counts[0]}, Real: {class_counts[1]}, scanned: {total_seen})")

        except Exception:
            errors += 1
            if errors > 50:
                print(f"  Too many errors ({errors}), stopping collection")
                break
            continue

    print(f"Collected {len(mels)} samples (Fake: {class_counts[0]}, Real: {class_counts[1]}, errors: {errors})")

    if len(mels) < 20:
        return None, None

    return np.array(mels), np.array(labels)


def load_dataset_local():
    """Load audio from local directories: data/audio/real/ and data/audio/fake/."""
    real_dir = os.path.join(ROOT_DIR, "data", "audio", "real")
    fake_dir = os.path.join(ROOT_DIR, "data", "audio", "fake")

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        return None, None

    mels = []
    labels = []
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

    for label, dirname in [(1, real_dir), (0, fake_dir)]:
        files = [
            os.path.join(dirname, f) for f in os.listdir(dirname)
            if os.path.splitext(f)[1].lower() in audio_exts
        ]
        print(f"Found {len(files)} {'real' if label == 1 else 'fake'} audio files")

        for fpath in tqdm(files, desc=f"Processing {'real' if label == 1 else 'fake'}"):
            try:
                waveform, sr = librosa.load(fpath, sr=SAMPLE_RATE, duration=MAX_DURATION)
                if len(waveform) < int(0.5 * SAMPLE_RATE):
                    continue
                mel = audio_to_mel(waveform, sr)
                mels.append(mel)
                labels.append(label)
            except Exception:
                continue

    if len(mels) == 0:
        return None, None

    return np.array(mels), np.array(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try local data first, then HuggingFace
    print("\nChecking for local audio data...")
    mels, labels = load_dataset_local()

    if mels is None or len(mels) < 20:
        print("Local data insufficient. Loading from HuggingFace...")
        mels, labels = load_dataset_hf()

    if mels is None or len(mels) < 20:
        print("\nError: Could not load sufficient training data.")
        print("Options:")
        print("  1. Place audio files in data/audio/real/ and data/audio/fake/")
        print("  2. Ensure HuggingFace datasets are accessible")
        return

    # Shuffle
    indices = list(range(len(labels)))
    random.seed(42)
    random.shuffle(indices)
    mels = mels[indices]
    labels = labels[indices]

    # Train/val split
    split = int(len(labels) * TRAIN_SPLIT)
    train_mels, val_mels = mels[:split], mels[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    print(f"\nTrain samples: {len(train_labels)}")
    print(f"Val samples  : {len(val_labels)}")
    print(f"Class balance: Fake={sum(labels == 0)}, Real={sum(labels == 1)}")

    train_loader = DataLoader(
        AudioMelDataset(train_mels, train_labels, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        AudioMelDataset(val_mels, val_labels, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )

    # Model
    model = AudioDeepfakeCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training loop
    print(f"\nStarting Audio Deepfake CNN training...\n")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for mels_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            mels_batch = mels_batch.to(device)
            labels_batch = labels_batch.to(device)

            preds = model(mels_batch)
            loss = criterion(preds, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for mels_batch, labels_batch in val_loader:
                mels_batch = mels_batch.to(device)
                labels_batch = labels_batch.to(device)

                preds = model(mels_batch)
                loss = criterion(preds, labels_batch)
                val_loss += loss.item()

                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == labels_batch).sum().item()
                total += labels_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"| Train Loss: {avg_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nAudio Deepfake CNN training complete.")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
