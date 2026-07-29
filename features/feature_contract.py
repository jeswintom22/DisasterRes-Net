"""Shared feature-pipeline contracts for training, evaluation, and inference.

The project historically used several names for closely related feature sets.
This module is the single source of truth for feature dimensions, names, and
artifact compatibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


CNN_ORIGINAL_DIM = 1000
CNN_SALIENCY_DIM = 1000
GLCM_DIM = 20
LBP_HIST_DIM = 10
LBP_STATS_DIM = 16


STREAM_DIMS: Dict[str, int] = {
    "cnn_original": CNN_ORIGINAL_DIM,
    "cnn_saliency": CNN_SALIENCY_DIM,
    "glcm": GLCM_DIM,
    "lbp_hist": LBP_HIST_DIM,
    "lbp_stats": LBP_STATS_DIM,
}

PIPELINE_ALIASES: Dict[str, List[str]] = {
    "cnn": ["cnn_original", "cnn_saliency"],
    "glcm": ["glcm"],
    "lbp": ["lbp_stats"],
    "lbp_hist": ["lbp_hist"],
    "cnn_lbp": ["cnn_original", "cnn_saliency", "lbp_stats"],
    "cnn_glcm": ["cnn_original", "cnn_saliency", "glcm"],
    "glcm_lbp": ["glcm", "lbp_stats"],
    "cnn_glcm_lbp": ["cnn_original", "cnn_saliency", "glcm", "lbp_stats"],
    "df1_df2_glcm": ["cnn_original", "cnn_saliency", "glcm"],
    "df1_df2_glcm_lbp": ["cnn_original", "cnn_saliency", "glcm", "lbp_hist"],
    "df1_df2_glcm_lbp_stats": ["cnn_original", "cnn_saliency", "glcm", "lbp_stats"],
}

LEGACY_PIPELINE_BY_DIM: Dict[int, str] = {
    2020: "df1_df2_glcm",
    2030: "df1_df2_glcm_lbp",
    2036: "df1_df2_glcm_lbp_stats",
}


@dataclass(frozen=True)
class FeaturePipelineSpec:
    """Feature stream contract used by scaler and classifier artifacts."""

    name: str
    streams: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return sum(STREAM_DIMS[stream] for stream in self.streams)

    @property
    def feature_names(self) -> List[str]:
        names: List[str] = []
        for stream in self.streams:
            names.extend(names_for_stream(stream))
        return names

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "streams": list(self.streams),
            "stream_dimensions": {stream: STREAM_DIMS[stream] for stream in self.streams},
            "dimension": self.dimension,
            "feature_count": len(self.feature_names),
        }


def names_for_stream(stream: str) -> List[str]:
    """Return deterministic publication-ready feature names for one stream."""
    if stream == "cnn_original":
        return [f"cnn_original_{idx + 1}" for idx in range(CNN_ORIGINAL_DIM)]
    if stream == "cnn_saliency":
        return [f"cnn_saliency_{idx + 1}" for idx in range(CNN_SALIENCY_DIM)]
    if stream == "glcm":
        angles = ["0", "45", "90", "135"]
        props = ["contrast", "correlation", "energy", "homogeneity", "entropy"]
        return [f"glcm_{prop}_{angle}" for angle in angles for prop in props]
    if stream == "lbp_hist":
        return [f"lbp_bin_{idx + 1}" for idx in range(LBP_HIST_DIM)]
    if stream == "lbp_stats":
        return [f"lbp_bin_{idx + 1}" for idx in range(LBP_HIST_DIM)] + [
            "lbp_mean",
            "lbp_variance",
            "lbp_entropy",
            "lbp_energy",
            "lbp_dominant_bin",
            "lbp_uniformity",
        ]
    raise ValueError(f"Unknown feature stream: {stream}")


def get_pipeline_spec(name: str) -> FeaturePipelineSpec:
    """Resolve a named pipeline configuration."""
    if name not in PIPELINE_ALIASES:
        valid = ", ".join(sorted(PIPELINE_ALIASES))
        raise ValueError(f"Unknown feature pipeline '{name}'. Valid options: {valid}")
    return FeaturePipelineSpec(name=name, streams=tuple(PIPELINE_ALIASES[name]))


def resolve_artifact_spec(requested_name: str, expected_dim: int | None) -> FeaturePipelineSpec:
    """Resolve an artifact's true feature contract from its scaler dimension.

    This protects backward compatibility with mislabeled artifacts, especially
    `df1_df2_glcm_lbp_stats` files whose scaler was fitted with 2030 legacy LBP
    features because an old cache was reused.
    """
    requested = get_pipeline_spec(requested_name)
    if expected_dim is None or requested.dimension == expected_dim:
        return requested
    if expected_dim in LEGACY_PIPELINE_BY_DIM:
        return get_pipeline_spec(LEGACY_PIPELINE_BY_DIM[expected_dim])
    raise ValueError(
        f"Artifact feature mismatch for '{requested_name}': requested dimension "
        f"{requested.dimension}, scaler expects {expected_dim}. No compatible "
        "DisasterRes-Net feature contract is registered for that dimension."
    )


def build_feature_matrix(
    available_features: Mapping[str, np.ndarray],
    spec: FeaturePipelineSpec,
) -> np.ndarray:
    """Concatenate feature arrays according to a spec with dimension validation."""
    parts: List[np.ndarray] = []
    missing = [stream for stream in spec.streams if stream not in available_features]
    if missing:
        raise ValueError(f"Missing feature streams for '{spec.name}': {missing}")

    row_count = None
    for stream in spec.streams:
        arr = np.asarray(available_features[stream], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        expected = STREAM_DIMS[stream]
        if arr.shape[1] != expected:
            raise ValueError(
                f"Feature stream '{stream}' has {arr.shape[1]} columns; expected {expected}."
            )
        row_count = arr.shape[0] if row_count is None else row_count
        if arr.shape[0] != row_count:
            raise ValueError(
                f"Feature stream '{stream}' has {arr.shape[0]} rows; expected {row_count}."
            )
        parts.append(arr)

    fused = np.hstack(parts).astype(np.float32)
    validate_feature_matrix(fused, spec)
    return fused


def validate_feature_matrix(matrix: np.ndarray, spec: FeaturePipelineSpec) -> None:
    """Raise an informative exception when a feature matrix violates its spec."""
    if matrix.ndim != 2:
        raise ValueError(f"Feature matrix for '{spec.name}' must be 2D, got shape {matrix.shape}.")
    if matrix.shape[1] != spec.dimension:
        raise ValueError(
            f"Feature matrix for '{spec.name}' has {matrix.shape[1]} columns; "
            f"expected {spec.dimension}. Streams={list(spec.streams)}"
        )


def validate_feature_importance_length(importances: Sequence[float], spec: FeaturePipelineSpec) -> None:
    """Ensure feature importance arrays align with generated feature names."""
    if len(importances) != len(spec.feature_names):
        raise ValueError(
            f"Feature importance length mismatch for '{spec.name}': got {len(importances)}, "
            f"expected {len(spec.feature_names)}."
        )


def known_pipeline_specs(names: Iterable[str] | None = None) -> List[FeaturePipelineSpec]:
    """Return all requested feature-pipeline specs."""
    selected = list(names) if names is not None else list(PIPELINE_ALIASES)
    return [get_pipeline_spec(name) for name in selected]
