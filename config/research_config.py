"""Research configuration defaults for DisasterRes-Net."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible experiment settings shared by training and evaluation."""

    random_seed: int = 42
    classifier: str = "random_forest"
    n_estimators: int = 300
    cv_folds: int = 5
    localization_backend: str = "sun_ica"
    ablation_configs: List[str] = field(
        default_factory=lambda: [
            "cnn",
            "glcm",
            "lbp",
            "cnn_lbp",
            "cnn_glcm",
            "glcm_lbp",
            "cnn_glcm_lbp",
        ]
    )
    disaster_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "earthquake": 70.0,
            "flood": 68.0,
            "wildfire": 72.0,
            "hurricane": 69.0,
            "landslide": 71.0,
            "default": 70.0,
        }
    )


DEFAULT_CONFIG = ExperimentConfig()
