"""Configurable localization backends for DisasterRes-Net M2.

Backends return normalized activation maps in [0, 1]. The SUN+ICA backend is the
paper-reproduction baseline; OpenCV saliency is the current lightweight backend;
Grad-CAM family backends use a PyTorch model when supplied or lazily load the
project's InceptionResNetV2 feature extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import numpy as np
from PIL import Image
from sklearn.decomposition import FastICA

from preprocessing.saliency import SaliencyAttention

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import timm
except Exception:  # pragma: no cover
    torch = None
    nn = None
    transforms = None
    timm = None


@dataclass(frozen=True)
class LocalizationResult:
    """Normalized localization output from one backend."""

    backend: str
    activation_map: np.ndarray
    confidence: float
    metadata: Dict[str, object]


class LocalizationBackend(Protocol):
    """Interface implemented by all localization backends."""

    name: str

    def localize(self, img_rgb: np.ndarray) -> LocalizationResult:
        """Return a normalized activation map for one RGB image."""


def normalize_map(arr: np.ndarray) -> np.ndarray:
    """Normalize activation arrays to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax <= amin:
        return np.ones_like(arr, dtype=np.float32) * 0.5
    return (arr - amin) / (amax - amin)


class SunIcaBackend:
    """Original-paper baseline approximation: SUN-style saliency plus ICA.

    The original paper does not provide implementation-level parameters. This
    implementation documents the deviation and uses LMS color opponency,
    intensity, and gradient contrast features followed by one-component FastICA.
    """

    name = "sun_ica"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def localize(self, img_rgb: np.ndarray) -> LocalizationResult:
        resized = img_rgb.astype(np.uint8)
        h, w = resized.shape[:2]
        rgb = resized.astype(np.float32) / 255.0
        lms = self._rgb_to_lms(rgb)
        gray = np.asarray(Image.fromarray(resized).convert("L"), dtype=np.float32) / 255.0
        gy, gx = np.gradient(gray)
        grad = normalize_map(np.sqrt(np.square(gx) + np.square(gy)))
        features = np.column_stack(
            [
                lms.reshape(-1, 3),
                gray.reshape(-1),
                grad.reshape(-1),
            ]
        )
        features = features - features.mean(axis=0, keepdims=True)
        try:
            ica = FastICA(n_components=1, random_state=self.random_state, whiten="unit-variance", max_iter=300)
            component = np.abs(ica.fit_transform(features).reshape(h, w))
        except Exception:
            component = grad
        sun_prior = normalize_map(0.55 * grad + 0.45 * np.std(lms, axis=2))
        activation = normalize_map(0.58 * normalize_map(component) + 0.42 * sun_prior)
        return LocalizationResult(
            backend=self.name,
            activation_map=activation,
            confidence=float(np.std(activation)),
            metadata={
                "paper_reproduction": True,
                "deviation": "SUN saliency is approximated from LMS/intensity/gradient contrast; ICA uses sklearn FastICA.",
                "ica_components": 1,
            },
        )

    @staticmethod
    def _rgb_to_lms(rgb: np.ndarray) -> np.ndarray:
        matrix = np.array(
            [
                [0.3811, 0.5783, 0.0402],
                [0.1967, 0.7244, 0.0782],
                [0.0241, 0.1288, 0.8444],
            ],
            dtype=np.float32,
        )
        return np.clip(rgb @ matrix.T, 0.0, 1.0)


class OpenCVSaliencyBackend:
    """Current OpenCV/static-saliency backend."""

    name = "opencv"

    def __init__(self) -> None:
        self.saliency = SaliencyAttention()

    def localize(self, img_rgb: np.ndarray) -> LocalizationResult:
        output = self.saliency.process(img_rgb)
        return LocalizationResult(
            backend=self.name,
            activation_map=output.saliency_map,
            confidence=float(np.std(output.saliency_map)),
            metadata={"method": output.method},
        )


class TorchCamBackend:
    """Grad-CAM, Grad-CAM++, and Score-CAM over InceptionResNetV2."""

    def __init__(self, mode: str = "gradcam", model: Optional[object] = None, target_layer: Optional[object] = None) -> None:
        if mode not in {"gradcam", "gradcam_plus_plus", "scorecam"}:
            raise ValueError(f"Unsupported CAM mode: {mode}")
        self.name = mode
        self.model = model
        self.target_layer = target_layer
        self._device = None

    def localize(self, img_rgb: np.ndarray) -> LocalizationResult:
        if torch is None or timm is None or transforms is None:
            raise RuntimeError("PyTorch, torchvision, and timm are required for CAM localization backends.")
        model = self.model or self._load_default_model()
        target_layer = self.target_layer or self._find_last_conv(model)
        activation, confidence = self._cam(model, target_layer, img_rgb)
        return LocalizationResult(
            backend=self.name,
            activation_map=activation,
            confidence=confidence,
            metadata={"target_layer": target_layer.__class__.__name__, "mode": self.name},
        )

    def _load_default_model(self):
        device = self._get_device()
        model = timm.create_model("inception_resnet_v2", pretrained=True, num_classes=1000)
        model.eval().to(device)
        return model

    def _get_device(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    @staticmethod
    def _find_last_conv(model):
        last_conv = None
        for module in model.modules():
            if nn is not None and isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found for CAM localization.")
        return last_conv

    def _prepare_tensor(self, img_rgb: np.ndarray):
        transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(Image.fromarray(img_rgb.astype(np.uint8))).unsqueeze(0).to(self._get_device())
        tensor.requires_grad_(self.name != "scorecam")
        return tensor

    def _cam(self, model, target_layer, img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        activations = []
        gradients = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)
        try:
            tensor = self._prepare_tensor(img_rgb)
            logits = model(tensor)
            score = logits.max(dim=1).values.sum()
            confidence = float(torch.softmax(logits, dim=1).max().detach().cpu().item())
            if self.name == "scorecam":
                cam = self._scorecam(model, tensor, activations[0], logits.argmax(dim=1).item())
            else:
                model.zero_grad(set_to_none=True)
                score.backward(retain_graph=True)
                act = activations[0].detach()
                grad = gradients[0].detach()
                if self.name == "gradcam_plus_plus":
                    weights = self._gradcam_plus_weights(act, grad)
                else:
                    weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
            cam_np = cam.squeeze().detach().float().cpu().numpy()
            if cv2 is not None:
                cam_np = cv2.resize(cam_np, (img_rgb.shape[1], img_rgb.shape[0]))
            else:
                cam_np = np.asarray(Image.fromarray(normalize_map(cam_np)).resize((img_rgb.shape[1], img_rgb.shape[0])))
            return normalize_map(cam_np), confidence
        finally:
            handle_f.remove()
            handle_b.remove()

    @staticmethod
    def _gradcam_plus_weights(activations, gradients):
        grads_power_2 = gradients.pow(2)
        grads_power_3 = gradients.pow(3)
        denom = 2 * grads_power_2 + (activations * grads_power_3).sum(dim=(2, 3), keepdim=True)
        alpha = grads_power_2 / torch.clamp(denom, min=1e-8)
        positive_grad = torch.relu(gradients)
        return (alpha * positive_grad).sum(dim=(2, 3), keepdim=True)

    def _scorecam(self, model, tensor, activations, target_idx: int):
        act = activations.detach()
        _, channels, _, _ = act.shape
        channel_scores = []
        max_channels = min(channels, 16)
        for idx in range(max_channels):
            mask = act[:, idx : idx + 1]
            mask = torch.nn.functional.interpolate(mask, size=tensor.shape[2:], mode="bilinear", align_corners=False)
            mask = (mask - mask.min()) / torch.clamp(mask.max() - mask.min(), min=1e-8)
            with torch.no_grad():
                score = torch.softmax(model(tensor * mask), dim=1)[0, target_idx]
            channel_scores.append(score)
        weights = torch.stack(channel_scores).reshape(1, max_channels, 1, 1)
        return torch.relu((weights * act[:, :max_channels]).sum(dim=1, keepdim=True))


def get_localization_backend(name: str, **kwargs) -> LocalizationBackend:
    """Factory for configured localization backends."""
    normalized = name.lower().replace("-", "_")
    if normalized in {"sun_ica", "sun+ica", "paper"}:
        return SunIcaBackend(**kwargs)
    if normalized in {"opencv", "opencv_saliency", "current"}:
        return OpenCVSaliencyBackend()
    if normalized in {"gradcam", "grad_cam"}:
        return TorchCamBackend("gradcam", **kwargs)
    if normalized in {"gradcam++", "grad_cam_plus_plus", "gradcam_plus_plus"}:
        return TorchCamBackend("gradcam_plus_plus", **kwargs)
    if normalized in {"scorecam", "score_cam"}:
        return TorchCamBackend("scorecam", **kwargs)
    raise ValueError(f"Unknown localization backend: {name}")
