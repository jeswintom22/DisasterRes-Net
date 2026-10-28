"""Phase 9 Prerequisite: Image Perspective Classifier Gate.

Classifies input disaster imagery into:
- 'overhead_or_aerial' (satellite, drone, or high-angle aerial views suitable for SegFormer-B2)
- 'ground_level' (street-level or eye-level photos routed to Grad-CAM++ / saliency DDM)
- 'graphic_or_screenshot' (infographics, maps, text memes flagged before segmentation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class PerspectiveVerdict:
    perspective: str
    recommended_localization_backend: str
    confidence: float
    metrics: Dict[str, float]


class PerspectiveGate:
    """Fast texture and vertical-gradient perspective classifier for M2 routing."""

    def classify(self, img_rgb: np.ndarray) -> PerspectiveVerdict:
        if img_rgb is None or img_rgb.size == 0:
            return PerspectiveVerdict("ground_level", "gradcam_plus_plus", 0.5, {})

        arr = np.asarray(img_rgb, dtype=np.float32)
        if arr.ndim == 2:
            gray = arr
        else:
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

        h, w = gray.shape[:2]
        # Compute flat pixel ratio (near-zero local gradient characteristic of text graphics/screenshots)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        grad_mag = np.hypot(gx, gy)
        flat_fraction = float(np.mean(grad_mag < 2.0))

        # Top-third vs bottom-third texture ratio (ground-level scenes often have smoother sky at top)
        top_third = grad_mag[: max(1, h // 3), :]
        bot_third = grad_mag[max(1, (2 * h) // 3) :, :]
        top_grad = float(np.mean(top_third)) + 1e-3
        bot_grad = float(np.mean(bot_third)) + 1e-3
        top_to_bottom_ratio = top_grad / bot_grad
        overall_grad = float(np.mean(grad_mag))

        metrics = {
            "flat_fraction": round(flat_fraction, 4),
            "top_to_bottom_grad_ratio": round(top_to_bottom_ratio, 4),
            "overall_grad_mean": round(overall_grad, 4),
        }

        if flat_fraction > 0.58:
            return PerspectiveVerdict(
                perspective="graphic_or_screenshot",
                recommended_localization_backend="saliency_fallback",
                confidence=round(min(0.95, 0.5 + (flat_fraction - 0.58)), 3),
                metrics=metrics,
            )

        # Aerial/overhead views exhibit uniform high-frequency texture across top and bottom thirds
        if 0.78 <= top_to_bottom_ratio <= 1.35 and overall_grad >= 14.0 and flat_fraction < 0.28:
            return PerspectiveVerdict(
                perspective="overhead_or_aerial",
                recommended_localization_backend="segformer_b2_aerial",
                confidence=0.78,
                metrics=metrics,
            )

        return PerspectiveVerdict(
            perspective="ground_level",
            recommended_localization_backend="gradcam_plus_plus",
            confidence=0.82,
            metrics=metrics,
        )

