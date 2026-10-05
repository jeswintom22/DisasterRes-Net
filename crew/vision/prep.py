from __future__ import annotations
import numpy as np
from PIL import Image

def resize_rgb_to_max_edge(image: np.ndarray, max_edge: int = 1280) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_edge:
        return image.astype(np.uint8, copy=False)
    scale = max_edge / max(height, width)
    resized = Image.fromarray(image.astype(np.uint8)).resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)
