import torch
import torch.nn as nn
import clip


class GeneralAIImageDetector(nn.Module):
    """
    CLIP-based General AI Image Detector
    - CLIP backbone is frozen
    - Small trainable classifier head
    - Handles FP16 (CLIP) → FP32 (classifier) safely
    """

    def __init__(self, device):
        super().__init__()

        # Load CLIP backbone
        self.clip_model, _ = clip.load("ViT-B/32", device=device)

        # Freeze CLIP weights
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Classification head (FP32)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        """
        image: Tensor [B, 3, 224, 224]
        returns: AI probability [B, 1]
        """

        # CLIP feature extraction (usually FP16 on GPU)
        with torch.no_grad():
            features = self.clip_model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)

        # 🔑 CRITICAL FIX:
        # Convert features to FP32 before classifier
        features = features.float()

        return self.classifier(features)
