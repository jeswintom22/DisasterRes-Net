"""Conservative image quality gate. Domain classifiers can be plugged in later."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str
    confidence_band: str

class ImageSuitabilityGate:
    def __init__(self) -> None:
        self._fingerprints: set[bytes] = set()

    def evaluate(self, image: np.ndarray) -> GateResult:
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 96:
            return GateResult(False, "Image is too small or is not RGB.", "none")
        fingerprint = image[::32, ::32].mean(axis=2).astype(np.uint8).tobytes()
        if fingerprint in self._fingerprints:
            return GateResult(False, "Duplicate image.", "none")
        self._fingerprints.add(fingerprint)
        if float(image.std()) < 8:
            return GateResult(False, "Image has insufficient visual detail.", "none")
        return GateResult(True, "Passed geometric and duplicate checks; overhead suitability is unverified.", "low")
