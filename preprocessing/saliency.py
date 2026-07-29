"""Saliency-guided attention preprocessing for DisasterRes-Net."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None


@dataclass(frozen=True)
class SaliencyOutput:
    """Saliency mask and saliency-weighted image used by the CNN stream."""

    saliency_map: np.ndarray
    attention_mask: np.ndarray
    enhanced_rgb: np.ndarray
    method: str


class SaliencyAttention:
    """Create a saliency attention mask that influences CNN inference."""

    def __init__(self, attention_strength: float = 0.55, blur_kernel: int = 7) -> None:
        self.attention_strength = float(np.clip(attention_strength, 0.0, 1.0))
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def process(self, img_rgb: np.ndarray) -> SaliencyOutput:
        saliency, method = self._detect_saliency(img_rgb)
        attention = self._make_attention_mask(saliency)
        enhanced = self.apply_attention(img_rgb, attention, self.attention_strength)
        return SaliencyOutput(saliency, attention, enhanced, method)

    def _detect_saliency(self, img_rgb: np.ndarray) -> tuple[np.ndarray, str]:
        if cv2 is not None:
            saliency = self._opencv_saliency(img_rgb)
            if saliency is not None:
                return saliency, "opencv_static_saliency"
            spectral = self._spectral_residual(img_rgb)
            if spectral is not None:
                return spectral, "spectral_residual"
        return self._laplacian_contrast(img_rgb), "laplacian_contrast"

    def _opencv_saliency(self, img_rgb: np.ndarray) -> np.ndarray | None:
        if not hasattr(cv2, "saliency"):
            return None
        try:
            detector = cv2.saliency.StaticSaliencyFineGrained_create()
            ok, saliency = detector.computeSaliency(img_rgb.astype(np.uint8))
            if ok:
                return self._normalize(saliency)
        except Exception:
            return None
        return None

    def _spectral_residual(self, img_rgb: np.ndarray) -> np.ndarray | None:
        try:
            gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            fft = np.fft.fft2(gray.astype(np.float32))
            amplitude = np.abs(fft)
            log_amplitude = np.log(amplitude + 1e-8)
            phase = np.angle(fft)
            avg_log = cv2.blur(log_amplitude, (3, 3))
            residual = log_amplitude - avg_log
            saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
            saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
            saliency = cv2.resize(saliency, (img_rgb.shape[1], img_rgb.shape[0]))
            return self._normalize(saliency)
        except Exception:
            return None

    def _laplacian_contrast(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = np.asarray(Image.fromarray(img_rgb.astype(np.uint8)).convert("L"), dtype=np.float32) / 255.0
        if cv2 is not None:
            saliency = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        else:
            gy, gx = np.gradient(gray)
            saliency = np.sqrt(np.square(gx) + np.square(gy))
        return self._normalize(saliency)

    def _make_attention_mask(self, saliency: np.ndarray) -> np.ndarray:
        attention = self._normalize(saliency)
        if cv2 is not None:
            attention = cv2.GaussianBlur(attention.astype(np.float32), (self.blur_kernel, self.blur_kernel), 0)
        return self._normalize(attention)

    @staticmethod
    def apply_attention(img_rgb: np.ndarray, attention: np.ndarray, strength: float) -> np.ndarray:
        image = img_rgb.astype(np.float32) / 255.0
        mask = attention.astype(np.float32)
        if mask.shape[:2] != image.shape[:2]:
            if cv2 is None:
                mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(image.shape[1::-1])) / 255.0
            else:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = mask[..., None]
        gain = 1.0 + strength * mask
        background = 1.0 - (strength * 0.35) * (1.0 - mask)
        enhanced = np.clip(image * gain * background, 0.0, 1.0)
        return (enhanced * 255).astype(np.uint8)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        arr_min = float(np.nanmin(arr))
        arr_max = float(np.nanmax(arr))
        if not np.isfinite(arr_min) or not np.isfinite(arr_max) or arr_max <= arr_min:
            return np.ones_like(arr, dtype=np.float32) * 0.5
        return (arr - arr_min) / (arr_max - arr_min)


def saliency_to_rgb(saliency_map: np.ndarray) -> np.ndarray:
    """Convert a saliency map to a three-channel uint8 image."""
    saliency = SaliencyAttention._normalize(saliency_map)
    return (np.repeat(saliency[..., None], 3, axis=2) * 255).astype(np.uint8)

