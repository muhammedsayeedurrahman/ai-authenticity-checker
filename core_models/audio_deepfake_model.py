"""
Audio Deepfake Detection CNN — PyTorch re-implementation of zo9999/deepfake-audio-detector.

Architecture (matches the original Keras model exactly):
  Input:  (batch, 1, 91, 150)  — single-channel mel-spectrogram
  Conv2D(32, 3x3, ReLU) -> MaxPool2D(2x2)
  Conv2D(64, 3x3, ReLU) -> MaxPool2D(2x2)
  Flatten -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(2, Softmax)

Output: (batch, 2) — [p_fake, p_real] softmax probabilities
"""

import torch
import torch.nn as nn


class AudioDeepfakeCNN(nn.Module):
    """
    CNN for classifying mel-spectrograms as real or AI-generated speech.

    Input shape:  (batch, 1, 91, 150)
    Output shape: (batch, 2)  — index 0 = P(fake), index 1 = P(real)
    """

    N_MELS = 91
    MAX_TIME_STEPS = 150

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: Conv(32) + MaxPool
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: Conv(64) + MaxPool
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After Conv2D(32,3x3) on (91,150): -> (89,148)
        # After MaxPool(2x2): -> (44, 74)
        # After Conv2D(64,3x3): -> (42, 72)
        # After MaxPool(2x2): -> (21, 36)
        # Flatten: 21 * 36 * 64 = 48384
        self._flat_size = 21 * 36 * 64  # 48384

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 91, 150) mel-spectrogram tensor

        Returns:
            (batch, 2) softmax probabilities [p_fake, p_real]
        """
        x = self.features(x)
        x = self.classifier(x)
        return torch.softmax(x, dim=1)

    def predict_fake_prob(self, x):
        """Convenience: return P(fake) as a scalar for single input."""
        probs = self.forward(x)
        return probs[:, 0]  # index 0 = fake probability
