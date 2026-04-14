"""
Frequency-domain CNN for deepfake detection.

Replaces the heuristic FrequencyAnalyzer with a trainable CNN
that operates on properly normalized FFT magnitude spectra.

Preprocessing fix:
  - FFT -> log(1 + abs(fft)) -> per-image normalize to [0, 1]
  - This fixes the scaling mismatch that caused near-zero scores

Architecture:
  Input: (batch, 1, 256, 256) — normalized FFT magnitude
  Conv2D(16, 3x3) -> ReLU -> MaxPool(2)
  Conv2D(32, 3x3) -> ReLU -> MaxPool(2)
  Conv2D(64, 3x3) -> ReLU -> AdaptiveAvgPool(4x4)
  Flatten -> Linear(1024, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 1) -> Sigmoid

Output: P(AI-generated)

Saves as: models/frequency.pth
"""

import torch
import torch.nn as nn
import numpy as np
import cv2


def compute_fft_magnitude(pil_image, size=256):
    """
    Compute properly normalized FFT magnitude spectrum from a PIL image.

    Args:
        pil_image: PIL Image (RGB or grayscale)
        size: Output spectrum size (square)

    Returns:
        numpy array (size, size) normalized to [0, 1]
    """
    img = np.array(pil_image.convert("L"), dtype=np.float32)
    img = cv2.resize(img, (size, size))

    # 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Log magnitude (fixes scaling issue)
    magnitude = np.log1p(np.abs(fshift))

    # Per-image normalization to [0, 1]
    mag_min = magnitude.min()
    mag_max = magnitude.max()
    if mag_max - mag_min > 1e-8:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)

    return magnitude


def fft_to_tensor(pil_image, size=256):
    """
    Convert PIL image to FFT magnitude tensor ready for the CNN.

    Returns:
        torch.Tensor of shape (1, size, size) — single channel
    """
    mag = compute_fft_magnitude(pil_image, size=size)
    return torch.from_numpy(mag).unsqueeze(0).float()


class FrequencyCNN(nn.Module):
    """
    CNN classifier on FFT magnitude spectra.
    Outputs P(AI-generated).
    """

    def __init__(self, input_size=256):
        super().__init__()
        self.input_size = input_size

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 256, 256) normalized FFT magnitude

        Returns:
            (batch, 1) P(AI-generated)
        """
        x = self.features(x)
        return self.classifier(x)
