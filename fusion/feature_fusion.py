"""Feature fusion utilities for hybrid CNN and handcrafted descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from features.feature_contract import (
    FeaturePipelineSpec,
    build_feature_matrix,
    get_pipeline_spec,
)


@dataclass(frozen=True)
class FusionMetadata:
    """Describes the active feature-fusion contract."""

    strategy: str
    pipeline: str
    streams: tuple[str, ...]
    stream_dimensions: Dict[str, int]
    fused_dim: int
    feature_names: tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "pipeline": self.pipeline,
            "streams": list(self.streams),
            "stream_dimensions": dict(self.stream_dimensions),
            "fused_dim": self.fused_dim,
            "feature_count": len(self.feature_names),
            "description": (
                "Configurable early fusion of CNN, GLCM, and LBP streams before "
                "StandardScaler plus RandomForest classification."
            ),
        }


def fuse_feature_streams(
    available_features: Mapping[str, np.ndarray],
    spec: FeaturePipelineSpec,
) -> tuple[np.ndarray, FusionMetadata]:
    """Fuse available streams according to one shared feature spec."""
    fused = build_feature_matrix(available_features, spec)
    metadata = FusionMetadata(
        strategy="configurable_early_concatenation",
        pipeline=spec.name,
        streams=spec.streams,
        stream_dimensions={stream: available_features[stream].reshape(available_features[stream].shape[0], -1).shape[1] for stream in spec.streams},
        fused_dim=int(fused.shape[1]),
        feature_names=tuple(spec.feature_names),
    )
    return fused, metadata


def concatenate_features(
    cnn_features: np.ndarray,
    saliency_features: np.ndarray,
    glcm_features: np.ndarray,
    lbp_features: np.ndarray | None,
) -> tuple[np.ndarray, FusionMetadata]:
    """Backward-compatible fusion wrapper.

    New code should call :func:`fuse_feature_streams` with an explicit
    `FeaturePipelineSpec`. This wrapper infers legacy-vs-stats LBP from the
    provided array width for old call sites.
    """
    available = {
        "cnn_original": cnn_features,
        "cnn_saliency": saliency_features,
        "glcm": glcm_features,
    }
    if lbp_features is None or not lbp_features.size:
        spec = get_pipeline_spec("df1_df2_glcm")
    else:
        lbp = np.asarray(lbp_features, dtype=np.float32)
        if lbp.ndim == 1:
            lbp = lbp.reshape(1, -1)
        if lbp.shape[1] == 10:
            available["lbp_hist"] = lbp
            spec = get_pipeline_spec("df1_df2_glcm_lbp")
        elif lbp.shape[1] == 16:
            available["lbp_stats"] = lbp
            spec = get_pipeline_spec("df1_df2_glcm_lbp_stats")
        else:
            raise ValueError(f"Unsupported LBP feature width: {lbp.shape[1]}")
    return fuse_feature_streams(available, spec)
