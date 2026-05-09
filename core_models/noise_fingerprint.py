"""
Noise fingerprint detector stub.

Analyzes residual noise patterns to detect AI-generated images.
Camera sensors leave unique noise fingerprints; AI generators produce
different noise distributions.

Not yet integrated into the main pipeline — structure only for future use.

Architecture:
    Input: (1, 256, 256) residual noise map
    -> Conv2d(1, 32, 3) -> BN -> ReLU
    -> Conv2d(32, 64, 3) -> BN -> ReLU
    -> Conv2d(64, 128, 3) -> BN -> ReLU
    -> GlobalAvgPool -> Linear(128, 64) -> ReLU
    -> Dropout(0.3) -> Linear(64, 1) -> Sigmoid

Input:  PIL Image (RGB) — noise map extracted internally
Output: P(AI-generated) in [0, 1]
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class NoiseFingerprint(nn.Module):
    """
    CNN operating on residual noise maps for deepfake detection.

    Residual noise is computed by subtracting a denoised version of the
    image from the original, exposing sensor noise patterns (or lack thereof
    in AI-generated images).
    """

    NOISE_SIZE = 256

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, noise_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise_map: (batch, 1, 256, 256) residual noise tensor

        Returns:
            (batch, 1) P(AI-generated)
        """
        features = self.features(noise_map)
        return self.classifier(features)

    @staticmethod
    def extract_noise_map(pil_image: Image.Image, size: int = 256) -> np.ndarray:
        """
        Extract residual noise map from a PIL image.

        Uses Gaussian blur as a simple denoiser. The residual
        (original - blurred) reveals noise patterns.

        Args:
            pil_image: RGB PIL Image
            size: output size (square)

        Returns:
            (size, size) float32 numpy array
        """
        import cv2

        img = np.array(pil_image.convert("L").resize((size, size))).astype(np.float32)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        noise = img - blurred
        return noise

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> float:
        """Single-image inference from PIL Image."""
        self.eval()
        device = next(self.parameters()).device
        noise = self.extract_noise_map(pil_image)
        tensor = torch.tensor(noise, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return self.forward(tensor).item()
