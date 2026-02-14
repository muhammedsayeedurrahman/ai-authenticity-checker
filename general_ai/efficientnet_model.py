import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class AIOriginCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights="DEFAULT")

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)
