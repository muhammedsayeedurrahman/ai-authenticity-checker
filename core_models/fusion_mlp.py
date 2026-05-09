"""
Learned Fusion MLP — replaces manual weighted averaging.

Takes calibrated per-model scores and learns optimal combination weights.
Includes per-model temperature calibration as learnable parameters.

Input:  [vit, efficientnet, forensic, frequency, face, dino, texture]
Output: P(AI-generated) after sigmoid

Saves as: models/fusion_mlp.pth
"""

import torch
import torch.nn as nn


class ModelCalibrator(nn.Module):
    """
    Learnable temperature scaling per model.
    Maps raw sigmoid/softmax outputs to calibrated probabilities.
    Each model gets its own temperature parameter.
    """

    def __init__(self, n_models=7):
        super().__init__()
        # Initialize temperatures to 1.0 (no scaling)
        self.temperatures = nn.Parameter(torch.ones(n_models))

    def forward(self, scores):
        """
        Args:
            scores: (batch, n_models) raw model outputs in [0, 1]

        Returns:
            (batch, n_models) calibrated scores
        """
        # Clamp to avoid log(0)
        scores = scores.clamp(0.001, 0.999)
        # Convert to logits
        logits = torch.log(scores / (1 - scores))
        # Apply per-model temperature
        calibrated_logits = logits / self.temperatures.clamp(min=0.1)
        return torch.sigmoid(calibrated_logits)


class FusionMLP(nn.Module):
    """
    Learned fusion network with integrated calibration.

    Architecture:
        calibrate -> Linear(7, 16) -> ReLU -> Dropout(0.2)
                  -> Linear(16, 8)  -> ReLU -> Dropout(0.1)
                  -> Linear(8, 1)   -> Sigmoid

    Input order: [vit, efficientnet, forensic, frequency, face, dino, texture]
    """

    MODEL_NAMES = [
        "vit", "efficientnet", "forensic", "frequency",
        "face", "dino", "texture",
    ]

    def __init__(self, n_inputs=7):
        super().__init__()

        self.calibrator = ModelCalibrator(n_models=n_inputs)

        self.fusion = nn.Sequential(
            nn.Linear(n_inputs, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, scores):
        """
        Args:
            scores: (batch, 7) tensor of model scores, each in [0, 1].

        Returns:
            (batch, 1) fused P(AI-generated)
        """
        calibrated = self.calibrator(scores)
        return self.fusion(calibrated)

    def predict(
        self,
        vit=0.0,
        efficientnet=0.0,
        forensic=0.0,
        frequency=0.0,
        face=0.0,
        dino=0.0,
        texture=0.0,
    ):
        """
        Convenience method for single-sample inference.

        Returns:
            float: fused risk score in [0, 1]
        """
        scores = torch.tensor(
            [[vit, efficientnet, forensic, frequency, face, dino, texture]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            return self.forward(scores).item()
