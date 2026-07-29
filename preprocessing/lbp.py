"""Local Binary Pattern feature extraction for DisasterRes-Net.

LBP is used as a second feature stream, not only as a visualization.  The
extractor returns a normalized histogram plus compact statistical descriptors
that can be concatenated with CNN features before classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import local_binary_pattern


LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_POINTS + 2


@dataclass(frozen=True)
class LBPFeatureSet:
    """Complete LBP output for one image."""

    histogram: np.ndarray
    statistics: Dict[str, float]
    feature_vector: np.ndarray
    texture_map: np.ndarray


class LBPFeatureExtractor:
    """Compute standardized texture descriptors from Local Binary Patterns."""

    def __init__(
        self,
        points: int = LBP_POINTS,
        radius: int = LBP_RADIUS,
        method: str = LBP_METHOD,
        image_size: tuple[int, int] = (299, 299),
    ) -> None:
        self.points = points
        self.radius = radius
        self.method = method
        self.image_size = image_size
        self.bin_count = points + 2

    def extract_from_path(self, image_path: str) -> LBPFeatureSet:
        """Load an image and extract its LBP feature stream."""
        try:
            with Image.open(image_path) as img:
                gray = img.convert("L").resize(self.image_size, Image.Resampling.LANCZOS)
                return self.extract_from_gray(np.asarray(gray, dtype=np.uint8))
        except (OSError, UnidentifiedImageError):
            return self.empty()

    def extract_from_rgb(self, img_rgb: np.ndarray) -> LBPFeatureSet:
        """Extract LBP features from an RGB array."""
        gray = Image.fromarray(img_rgb.astype(np.uint8)).convert("L")
        gray = gray.resize(self.image_size, Image.Resampling.LANCZOS)
        return self.extract_from_gray(np.asarray(gray, dtype=np.uint8))

    def extract_from_gray(self, gray: np.ndarray) -> LBPFeatureSet:
        """Convert grayscale pixels into histogram and statistics."""
        lbp = local_binary_pattern(gray, self.points, self.radius, method=self.method)
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.bin_count + 1),
            range=(0, self.bin_count),
            density=False,
        )
        hist = hist.astype(np.float32)
        hist = hist / max(float(hist.sum()), 1.0)

        stats = self._statistics(hist)
        stat_vector = np.asarray(
            [
                stats["mean"],
                stats["variance"],
                stats["entropy"],
                stats["energy"],
                stats["dominant_bin"],
                stats["uniformity"],
            ],
            dtype=np.float32,
        )
        feature_vector = np.concatenate([hist, stat_vector]).astype(np.float32)
        texture_map = self.normalize_texture_map(lbp)
        return LBPFeatureSet(hist, stats, feature_vector, texture_map)

    def empty(self) -> LBPFeatureSet:
        """Return a shape-stable zero feature set for unreadable images."""
        hist = np.zeros(self.bin_count, dtype=np.float32)
        stats = {
            "mean": 0.0,
            "variance": 0.0,
            "entropy": 0.0,
            "energy": 0.0,
            "dominant_bin": 0.0,
            "uniformity": 0.0,
        }
        return LBPFeatureSet(
            histogram=hist,
            statistics=stats,
            feature_vector=np.concatenate([hist, np.zeros(6, dtype=np.float32)]),
            texture_map=np.zeros(self.image_size[::-1], dtype=np.float32),
        )

    def _statistics(self, hist: np.ndarray) -> Dict[str, float]:
        bins = np.arange(hist.size, dtype=np.float32)
        mean = float(np.sum(hist * bins))
        variance = float(np.sum(hist * np.square(bins - mean)))
        entropy = float(-np.sum(hist * np.log2(np.clip(hist, 1e-10, 1.0))))
        energy = float(np.sum(np.square(hist)))
        dominant_bin = float(np.argmax(hist) / max(hist.size - 1, 1))
        uniformity = float(np.sum(hist[: self.points + 1]))
        return {
            "mean": mean,
            "variance": variance,
            "entropy": entropy,
            "energy": energy,
            "dominant_bin": dominant_bin,
            "uniformity": uniformity,
        }

    @staticmethod
    def normalize_texture_map(lbp: np.ndarray) -> np.ndarray:
        """Normalize an LBP image for dashboard visualization."""
        lbp = lbp.astype(np.float32)
        lbp_min = float(lbp.min())
        lbp_max = float(lbp.max())
        if lbp_max <= lbp_min:
            return np.zeros_like(lbp, dtype=np.float32)
        return (lbp - lbp_min) / (lbp_max - lbp_min)

