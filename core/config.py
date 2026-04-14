"""
Configuration loader for ProofyX.

Reads configs/models.json and provides typed access to model registry,
ensemble weights, and calibration parameters.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env before anything reads os.environ
load_dotenv(override=False)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "models.json"
MODELS_DIR_DEFAULT = ROOT_DIR / "models"


@dataclass(frozen=True)
class ModelEntry:
    """Single model configuration entry."""
    name: str
    type: str  # "local" or "huggingface"
    enabled: bool = True
    path: Optional[str] = None
    model_id: Optional[str] = None  # HuggingFace model ID
    model_class: Optional[str] = None
    weight: Optional[float] = None
    description: str = ""
    requires_face: bool = False
    fast_mode: bool = False
    n_inputs: Optional[int] = None

    @property
    def full_path(self) -> Optional[Path]:
        if self.path is None:
            return None
        return MODELS_DIR_DEFAULT / self.path

    @property
    def exists(self) -> bool:
        fp = self.full_path
        return fp is not None and fp.exists()


@dataclass(frozen=True)
class CalibrationConfig:
    """Score calibration parameters."""
    temperature: float = 1.2
    high_confidence_override: float = 0.60


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""
    models_dir: Path = MODELS_DIR_DEFAULT
    device: str = "cpu"
    threshold: float = 0.5
    idle_unload_seconds: int = 600

    models: dict[str, ModelEntry] = field(default_factory=dict)
    ensemble_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def get_model(self, name: str) -> Optional[ModelEntry]:
        return self.models.get(name)

    def enabled_models(self) -> dict[str, ModelEntry]:
        return {k: v for k, v in self.models.items() if v.enabled}

    def get_weights(self, face_boosted: bool = False) -> dict[str, float]:
        key = "face_boosted" if face_boosted else "default"
        return dict(self.ensemble_weights.get(key, {}))


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from JSON file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    models_dir = ROOT_DIR / raw.get("models_dir", "models")

    # Parse model entries
    model_keys = [
        "vit", "texture", "frequency", "face", "dino",
        "efficientnet", "fusion", "corefakenet", "audio",
    ]
    models: dict[str, ModelEntry] = {}
    for key in model_keys:
        entry_raw = raw.get(key)
        if entry_raw is None:
            continue
        models[key] = ModelEntry(
            name=key,
            type=entry_raw.get("type", "local"),
            enabled=entry_raw.get("enabled", True),
            path=entry_raw.get("path"),
            model_id=entry_raw.get("model_id"),
            model_class=entry_raw.get("class"),
            weight=entry_raw.get("weight"),
            description=entry_raw.get("description", ""),
            requires_face=entry_raw.get("requires_face", False),
            fast_mode=entry_raw.get("fast_mode", False),
            n_inputs=entry_raw.get("n_inputs"),
        )

    calibration_raw = raw.get("calibration", {})
    calibration = CalibrationConfig(
        temperature=calibration_raw.get("temperature", 1.2),
        high_confidence_override=calibration_raw.get("high_confidence_override", 0.60),
    )

    return AppConfig(
        models_dir=models_dir,
        device=raw.get("device", "cpu"),
        threshold=raw.get("threshold", 0.5),
        idle_unload_seconds=raw.get("idle_unload_seconds", 600),
        models=models,
        ensemble_weights=raw.get("ensemble_weights", {}),
        calibration=calibration,
    )


# Module-level singleton — loaded once, reused everywhere
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
