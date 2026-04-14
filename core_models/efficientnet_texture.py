"""
EfficientNet-B4 Texture Branch for deepfake detection.

Purpose: Capture texture-level artifacts that complement ViT's structural analysis.
Trained on face-aligned crops to focus on skin texture, hair detail, and
boundary artifacts common in AI-generated portraits.

Architecture:
  EfficientNet-B4 (pretrained ImageNet) with:
    - Frozen early layers (features[0:5])
    - Unfrozen late layers (features[5:]) for texture-level fine-tuning
    - Custom classifier head: Linear(1792, 512) -> ReLU -> Dropout -> Linear(512, 1) -> Sigmoid

Output: P(AI-generated)

Saves as: models/efficient.pth
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EfficientNetTexture(nn.Module):
    """
    EfficientNet-B4 fine-tuned for detecting texture artifacts in deepfakes.
    Operates on face-aligned 224x224 crops.
    Outputs P(AI-generated).
    """

    def __init__(self):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        base = efficientnet_b4(weights=weights)

        # Freeze early feature blocks (low-level features are transferable)
        for i, block in enumerate(base.features):
            if i < 5:
                for p in block.parameters():
                    p.requires_grad = False

        self.features = base.features
        self.avgpool = base.avgpool

        # EfficientNet-B4 outputs 1792 features
        in_features = 1792
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224) normalized face crop

        Returns:
            (batch, 1) P(AI-generated)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)
