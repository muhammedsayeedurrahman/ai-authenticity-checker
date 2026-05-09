"""
CLIP ViT-L/14 deepfake detector.

Wraps the TorchScript model from `yermandy/deepfake-detection`
(CLIP ViT-L/14 with LN-tuning, 96.62% AUROC on CelebDF-v2, MIT license).

Uses hf_hub_download for the TorchScript model file and CLIPProcessor
for preprocessing. The TorchScript model is self-contained — no need
for CLIPModel at inference time.

Input:  PIL Image (RGB)
Output: float — P(fake) in [0, 1]
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ID = "yermandy/deepfake-detection"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


class CLIPDeepfakeDetector:
    """TorchScript-based CLIP ViT-L/14 deepfake detector."""

    REPO_ID = REPO_ID
    CLIP_MODEL = CLIP_MODEL_ID

    def __init__(self, device: torch.device, cache_dir: Optional[str] = None):
        self.device = device

        from huggingface_hub import hf_hub_download
        from transformers import CLIPProcessor

        # Download TorchScript model
        model_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename="model.torchscript",
            cache_dir=cache_dir,
        )

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL)

        logger.info(
            "CLIPDeepfakeDetector loaded: %s (processor: %s)",
            self.REPO_ID,
            self.CLIP_MODEL,
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        """Run inference on a single PIL image.

        Args:
            image: PIL Image (RGB).

        Returns:
            float — P(fake) in [0, 1].
        """
        inputs = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = self.model(pixel_values)
        probs = torch.softmax(logits, dim=1)

        # Column 1 = P(fake)
        fake_prob = probs[0, 1].item()
        return float(fake_prob)
