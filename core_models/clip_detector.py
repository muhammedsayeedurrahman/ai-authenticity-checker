"""
CLIP-based deepfake detector stub.

Uses CLIP image embeddings to detect AI-generated images via a linear probe.
Not yet integrated into the main pipeline — structure only for future use.

Architecture:
    CLIP ViT-B/32 embeddings (512-d) -> Linear(512, 256) -> ReLU
    -> Dropout(0.3) -> Linear(256, 1) -> Sigmoid

Input:  PIL Image (RGB)
Output: P(AI-generated) in [0, 1]
"""

import torch
import torch.nn as nn


class CLIPDetector(nn.Module):
    """
    Linear probe on frozen CLIP embeddings for deepfake detection.

    The CLIP backbone is loaded separately and kept frozen.
    Only the probe head is trainable.
    """

    EMBEDDING_DIM = 512

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, 512) CLIP image embeddings

        Returns:
            (batch, 1) P(AI-generated)
        """
        return self.probe(embeddings)

    @torch.no_grad()
    def predict_from_embedding(self, embedding: torch.Tensor) -> float:
        """Single-sample inference from a pre-computed embedding."""
        self.eval()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return self.forward(embedding).item()
