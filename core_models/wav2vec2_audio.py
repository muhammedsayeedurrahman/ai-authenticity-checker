"""
Wav2Vec2-based audio deepfake detector.

Wraps the HuggingFace model `garystafford/wav2vec2-deepfake-voice-detector`
(Wav2Vec2-XLSR-300M fine-tuned for audio deepfake detection, 97.9% accuracy).

Input:  16 kHz mono numpy array (float32)
Output: dict with fake_probability, confidence, label
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
SAMPLE_RATE = 16000


class Wav2Vec2AudioDetector:
    """HuggingFace Wav2Vec2 audio deepfake detector."""

    MODEL_ID = DEFAULT_MODEL_ID
    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, device: torch.device, model_id: Optional[str] = None):
        self.device = device
        model_id = model_id or self.MODEL_ID

        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioClassification,
        )

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        # Build label-to-index mapping for fake detection
        self._fake_idx = self._resolve_fake_index()

        logger.info("Wav2Vec2AudioDetector loaded: %s", model_id)

    def _resolve_fake_index(self) -> int:
        """Find the logit index that corresponds to 'fake' / 'spoof'."""
        id2label = getattr(self.model.config, "id2label", {})
        for idx, label in id2label.items():
            low = label.lower()
            if "fake" in low or "spoof" in low:
                return int(idx)
        # Fallback: assume index 0 = fake (matches garystafford model)
        return 0

    @torch.no_grad()
    def predict(self, waveform_16k: np.ndarray) -> dict:
        """Run inference on a 16 kHz mono waveform.

        Args:
            waveform_16k: 1-D float32 numpy array at 16 kHz.

        Returns:
            dict with keys: fake_probability, confidence, label
        """
        inputs = self.feature_extractor(
            waveform_16k,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        fake_prob = probs[self._fake_idx].item()
        confidence = abs(fake_prob - 0.5) * 2.0  # 0 = uncertain, 1 = confident

        if fake_prob > 0.7:
            label = "Likely Fake"
        elif fake_prob > 0.5:
            label = "Possibly Fake"
        elif fake_prob > 0.3:
            label = "Uncertain"
        else:
            label = "Likely Real"

        return {
            "fake_probability": float(fake_prob),
            "confidence": float(confidence),
            "label": label,
        }
