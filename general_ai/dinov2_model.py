import torch
import torch.nn as nn
from transformers import AutoModel

class DINOv2ImageDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        for p in self.backbone.parameters():
            p.requires_grad = False

        dim = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward_logits(self, x):
        out = self.backbone(pixel_values=x)
        features = out.last_hidden_state[:, 0]
        return self.classifier(features).squeeze(1)

    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x))
