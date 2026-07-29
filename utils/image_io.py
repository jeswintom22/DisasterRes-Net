"""Image encoding helpers for the Flask dashboard."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def image_to_data_url(img: np.ndarray, fmt: str = "JPEG", quality: int = 88) -> str:
    """Encode a numpy image as a browser-ready data URL."""
    arr = img
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L")
    else:
        pil = Image.fromarray(arr.astype(np.uint8))
    buffer = io.BytesIO()
    pil.save(buffer, format=fmt, quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{encoded}"


def heatmap_overlay(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    """Overlay a heatmap on an RGB image."""
    if cv2 is None:
        heat = np.repeat(heatmap[..., None], 3, axis=2)
        return np.clip((1.0 - alpha) * img_rgb + alpha * heat * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap.astype(np.float32), (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(heatmap_rgb, alpha, img_rgb.astype(np.uint8), 1 - alpha, 0)

