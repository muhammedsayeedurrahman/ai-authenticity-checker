import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2AuthModel(nn.Module):
    """
    Outputs P(AI-generated)
    """

    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        # Freeze all backbone parameters first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last 2 transformer blocks for fine-tuning
        for block in self.backbone.encoder.layer[-2:]:
            for p in block.parameters():
                p.requires_grad = True

        hidden = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)
