import torch
import torch.nn as nn
from torchvision import models


class FaceDeepfakeModel(nn.Module):
    """
    Outputs P(REAL)
    Uses ResNet50 backbone with partial unfreezing.
    """

    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all parameters first
        for p in base.parameters():
            p.requires_grad = False

        # Unfreeze layer4 for fine-tuning
        for p in base.layer4.parameters():
            p.requires_grad = True

        self.features = nn.Sequential(*list(base.children())[:-1])

        # ResNet50 outputs 2048 features (vs 512 for ResNet18)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
