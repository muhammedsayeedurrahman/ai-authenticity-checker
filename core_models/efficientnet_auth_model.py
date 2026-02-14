import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNetAuthModel(nn.Module):
    """
    Outputs P(AI-generated)
    """

    def __init__(self):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.model = efficientnet_v2_s(weights=weights)

        # Freeze all parameters first
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze last 2 feature blocks for fine-tuning
        for block in self.model.features[-2:]:
            for p in block.parameters():
                p.requires_grad = True

        in_f = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
