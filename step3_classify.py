"""Step 3: extract M1 features and train the paper-aligned classifier.

Architecture:
  L1: IRv2 on original image -> DF1 (1000-dim ImageNet logits)
  L2: IRv2 on saliency map   -> DF2 (1000-dim ImageNet logits)
    L3: GLCM handcrafted       -> HF  (20-dim)
    L4: LBP texture histogram   -> LT  (10-dim)
    Fusion: [DF1 | DF2 | HF | LT] = 2030-dim
  Classifier: Random Forest

This module is now PyTorch-first for all deep-model work while preserving the
existing data split, preprocessing expectations, feature fusion, evaluation
logic, and downstream artifact layout.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

from preprocessing.lbp import LBPFeatureExtractor, LBP_BINS
from preprocessing.saliency import SaliencyAttention, saliency_to_rgb as saliency_map_to_rgb
from features.feature_contract import (
    get_pipeline_spec,
    resolve_artifact_spec,
    validate_feature_importance_length,
)

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    print(f"[WARN] matplotlib import failed: {exc}. Confusion image generation will be disabled.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    import timm
except Exception as exc:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    transforms = None
    timm = None
    print(f"[ERROR] PyTorch / timm import failed: {exc}")
    print("[ERROR] Install torch, torchvision, and timm for Python 3.10 (Windows), then retry.")

try:
    import mlflow.pytorch
except Exception:
    pass


def npath(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


ROOT_DIR = npath(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = npath(ROOT_DIR, "split_dataset")
MODEL_DIR = npath(ROOT_DIR, "saved_models")
RESULT_DIR = npath(ROOT_DIR, "results")
GLCM_CACHE_DIR = npath(ROOT_DIR, ".cache")
FEATURE_CACHE_DIR = npath(GLCM_CACHE_DIR, "feature_cache")
IMG_SIZE = (299, 299)
BATCH_SIZE = 64
NUM_WORKERS = 0  # Windows-safe default

OBJECTIVES = ("informativeness", "damage")
IMG_EXTS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
FEATURE_PIPELINE = "df1_df2_glcm_lbp_stats"
LEGACY_FEATURE_PIPELINE = "df1_df2_glcm_lbp"
LEGACY_LBP_DIM = LBP_POINTS + 2
LBP_FEATURE_DIM = LBP_BINS + 6

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GLCM_CACHE_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)


DEFAULT_LABEL_MAPS = {
    "informativeness": {
        "earthquake": "informative",
        "flood": "informative",
        "hurricane": "informative",
        "wildfire": "informative",
        "landslide": "informative",
        "not_disaster": "not_informative",
    },
    "damage": {
        "earthquake": "severe",
        "hurricane": "severe",
        "flood": "mild",
        "wildfire": "mild",
        "landslide": "little_or_none",
        "not_disaster": "little_or_none",
    },
}


@dataclass
class TorchTrainingConfig:
    enabled: bool = os.getenv("DISASTERRES_TORCH_TRAIN", "0") == "1"
    epochs: int = int(os.getenv("DISASTERRES_TORCH_EPOCHS", "3"))
    batch_size: int = int(os.getenv("DISASTERRES_TORCH_BATCH_SIZE", str(BATCH_SIZE)))
    learning_rate: float = float(os.getenv("DISASTERRES_TORCH_LR", "3e-4"))
    weight_decay: float = float(os.getenv("DISASTERRES_TORCH_WEIGHT_DECAY", "1e-4"))
    val_fraction: float = float(os.getenv("DISASTERRES_TORCH_VAL_FRACTION", "0.15"))
    random_state: int = int(os.getenv("DISASTERRES_TORCH_SEED", "42"))
    label_smoothing: float = float(os.getenv("DISASTERRES_TORCH_LABEL_SMOOTHING", "0.05"))


_device = None
_shared_feature_models: Dict[str, torch.nn.Module] = {}
_saliency_attention = SaliencyAttention()
_lbp_extractor = LBPFeatureExtractor()


def get_device():
    global _device
    if _device is None and torch is not None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _device.type == "cuda":
            print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] No GPU found - running on CPU (slower).")
    return _device


def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError as exc:
        print(f"[ERROR] Could not list directory '{path}': {exc}")
        return []


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)


def get_transforms(train: bool = False):
    if transforms is None:
        raise RuntimeError("torchvision transforms are unavailable")

    ops = [transforms.Resize(IMG_SIZE)]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


class DisasterDataset(Dataset):
    def __init__(self, paths, labels, transform=None, saliency_input: bool = False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
        self.saliency_input = saliency_input

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            with Image.open(path) as img:
                img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            if self.saliency_input:
                sal = simple_saliency_map(img_rgb)
                img_pil = Image.fromarray(saliency_to_rgb(sal))
            else:
                img_pil = Image.fromarray(img_rgb)

            if self.transform is not None:
                img_tensor = self.transform(img_pil)
            else:
                img_tensor = img_pil
            return img_tensor, label
        except Exception:
            return torch.zeros((3, 299, 299), dtype=torch.float32), label


def rgb_to_lms(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to LMS color space (paper step 1 of SUN model)."""
    img = img_rgb.astype(np.float32) / 255.0
    rgb_to_lms_matrix = np.array(
        [
            [0.3, 0.622, 0.078],
            [0.23, 0.692, 0.078],
            [0.24342268924547819, 0.20476744424496821, 0.55314986650955360],
        ]
    )
    h, w = img_rgb.shape[:2]
    img_flat = img.reshape(-1, 3)
    lms_flat = img_flat @ rgb_to_lms_matrix.T
    lms = lms_flat.reshape(h, w, 3)
    return np.clip(lms, 0, 1)


def simple_saliency_map(img_rgb: np.ndarray) -> np.ndarray:
    """Generate saliency used by L2 and saliency-attention preprocessing."""
    try:
        return _saliency_attention.process(img_rgb).saliency_map
    except Exception as exc:
        print(f"[WARN] Saliency extraction failed: {exc}. Returning uniform saliency.")
        return np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32) * 0.5

def gradient_saliency_map(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Gradient-based saliency for one image tensor."""
    model.eval()
    inp = img_tensor.unsqueeze(0).to(device)
    if device.type == "cuda":
        inp = inp.half()
    inp = inp.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    output = model(inp)
    score = output[0].max()
    score.backward()

    sal = inp.grad.detach().abs().squeeze(0)
    sal = sal.max(dim=0).values
    sal = sal.float().cpu().numpy()

    sal_min, sal_max = sal.min(), sal.max()
    if sal_max > sal_min:
        sal = (sal - sal_min) / (sal_max - sal_min)
    else:
        sal = np.zeros_like(sal)
    return sal


def saliency_to_rgb(saliency_map: np.ndarray) -> np.ndarray:
    """Compatibility wrapper around the shared saliency visualization helper."""
    return saliency_map_to_rgb(saliency_map)

def _make_loader(paths, labels, train: bool = False, saliency_input: bool = False, batch_size: int = BATCH_SIZE):
    dataset = DisasterDataset(
        paths=paths,
        labels=labels,
        transform=get_transforms(train=train),
        saliency_input=saliency_input,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=get_device().type == "cuda",
    )


def build_timm_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    if timm is None or torch is None:
        raise RuntimeError("PyTorch / timm not available")
    return timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=num_classes)


def load_frozen_model(num_classes: int = 1000) -> Optional[torch.nn.Module]:
    """Load frozen pretrained Inception-ResNet-V2 for L1/L2 feature extraction."""
    if timm is None or torch is None:
        print("[ERROR] PyTorch / timm not available")
        return None

    try:
        model = build_timm_model(num_classes=num_classes, pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        device = get_device()
        model = model.to(device)
        if device.type == "cuda":
            model = model.half()
            print("[INFO] Using fp16 inference on GPU")
        model.eval()
        print("[INFO] Loaded frozen pretrained Inception-ResNet-V2")
        return model
    except Exception as exc:
        print(f"[ERROR] Failed to load frozen model: {exc}")
        return None


def _train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_seen += int(labels.size(0))

    if total_seen == 0:
        return 0.0, 0.0
    return total_loss / total_seen, total_correct / total_seen


def _evaluate_one_epoch(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_seen += int(labels.size(0))

    if total_seen == 0:
        return 0.0, 0.0
    return total_loss / total_seen, total_correct / total_seen


def train_pytorch_backbone(
    objective: str,
    train_paths: Sequence[str],
    y_train: np.ndarray,
    label_encoder: LabelEncoder,
    cfg: TorchTrainingConfig,
) -> Optional[Dict[str, float]]:
    """Optional PyTorch classifier training for direct-image inference and MLflow logging.

    This does not replace the paper-aligned RF fusion path. It is an incremental,
    parallel artifact that keeps the migrated project capable of pure-PyTorch
    training/inference when needed.
    """
    if not cfg.enabled:
        return None
    if torch is None or timm is None:
        print("[WARN] Skipping PyTorch classifier training because torch/timm are unavailable.")
        return None
    if len(train_paths) < 4:
        print("[WARN] Not enough images to create a validation split for PyTorch training.")
        return None

    device = get_device()
    val_fraction = min(max(cfg.val_fraction, 0.1), 0.4)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_paths)),
        test_size=val_fraction,
        stratify=y_train,
        random_state=cfg.random_state,
    )

    tr_paths = [train_paths[i] for i in train_idx]
    va_paths = [train_paths[i] for i in val_idx]
    tr_labels = y_train[train_idx]
    va_labels = y_train[val_idx]

    train_loader = _make_loader(tr_paths, tr_labels, train=True, batch_size=cfg.batch_size)
    val_loader = _make_loader(va_paths, va_labels, train=False, batch_size=cfg.batch_size)

    model = build_timm_model(num_classes=len(label_encoder.classes_), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))

    best_state = None
    best_val_acc = -1.0
    history: List[Dict[str, float]] = []

    print(f"[STEP] Optional PyTorch backbone training for '{objective}' ({cfg.epochs} epochs)")
    for epoch in range(cfg.epochs):
        train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = _evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_metrics)
        print(
            f"  [epoch {epoch + 1}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        return None

    model.load_state_dict(best_state)
    out_path = npath(MODEL_DIR, f"model_{objective}.pth")
    torch.save(model.state_dict(), out_path)

    history_path = npath(RESULT_DIR, f"torch_history_{objective}.json")
    with open(history_path, "w", encoding="utf-8") as fobj:
        json.dump({"objective": objective, "history": history, "config": asdict(cfg)}, fobj, indent=2)

    mlflow.log_params({f"torch_{k}": v for k, v in asdict(cfg).items()})
    best_metrics = history[-1].copy()
    best_metrics["best_val_acc"] = best_val_acc
    mlflow.log_metrics({f"torch_{k}": float(v) for k, v in best_metrics.items() if k != "epoch"})
    mlflow.log_artifact(out_path)
    mlflow.log_artifact(history_path)
    if hasattr(mlflow, "pytorch"):
        try:
            mlflow.pytorch.log_model(model, artifact_path=f"torch_model_{objective}")
        except Exception as exc:
            print(f"[WARN] MLflow PyTorch model logging failed: {exc}")

    print(f"[INFO] Saved optional PyTorch classifier checkpoint: {out_path}")
    return {
        "best_val_acc": best_val_acc,
        "epochs": float(cfg.epochs),
        "train_images": float(len(tr_paths)),
        "val_images": float(len(va_paths)),
    }


def extract_features_predictions_layer(
    model,
    image_paths: Sequence[str],
    batch_size: int = BATCH_SIZE,
    saliency_input: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract 1000-dim features from the IRv2 predictions layer."""
    if model is None:
        print("[ERROR] Model is None, cannot extract features")
        return np.zeros((len(image_paths), 1000), dtype=np.float32)

    model.eval()
    transform = get_transforms(train=False)
    device = get_device()
    predictions_features = []

    def hook_fn(module, inputs, output):
        predictions_features.append(output.detach().cpu().numpy())

    handle = model.classif.register_forward_hook(hook_fn)
    features = []
    try:
        for i in tqdm(range(0, len(image_paths), batch_size), desc="  L1/L2 features", disable=not show_progress):
            batch_paths = image_paths[i : i + batch_size]
            imgs_batch = []

            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
                    if saliency_input:
                        img_pil = Image.fromarray(saliency_to_rgb(simple_saliency_map(img_rgb)))
                    else:
                        img_pil = Image.fromarray(img_rgb)
                    imgs_batch.append(transform(img_pil))
                except Exception as exc:
                    print(f"[WARN] Failed to load '{path}': {exc}. Using zero tensor.")
                    imgs_batch.append(torch.zeros((3, 299, 299), dtype=torch.float32))

            if imgs_batch:
                imgs_tensor = torch.stack(imgs_batch).to(device)
                if device.type == "cuda":
                    imgs_tensor = imgs_tensor.half()
                predictions_features.clear()
                with torch.no_grad():
                    _ = model(imgs_tensor)
                if predictions_features:
                    features.append(predictions_features[0])
    finally:
        handle.remove()

    if features:
        return np.vstack(features)
    return np.zeros((len(image_paths), 1000), dtype=np.float32)


def extract_features_predictions_from_arrays(
    model,
    images_rgb: Sequence[np.ndarray],
    batch_size: int = BATCH_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    """Extract 1000-dim features from in-memory RGB arrays for web inference."""
    if model is None:
        print("[ERROR] Model is None, cannot extract features")
        return np.zeros((len(images_rgb), 1000), dtype=np.float32)

    model.eval()
    transform = get_transforms(train=False)
    device = get_device()
    predictions_features = []

    def hook_fn(module, inputs, output):
        predictions_features.append(output.detach().cpu().numpy())

    handle = model.classif.register_forward_hook(hook_fn)
    features = []
    try:
        for i in tqdm(range(0, len(images_rgb), batch_size), desc="  array CNN features", disable=not show_progress):
            batch_images = images_rgb[i : i + batch_size]
            imgs_batch = []

            for img_rgb in batch_images:
                try:
                    img_pil = Image.fromarray(np.asarray(img_rgb, dtype=np.uint8))
                    imgs_batch.append(transform(img_pil))
                except Exception as exc:
                    print(f"[WARN] Failed to transform in-memory image: {exc}. Using zero tensor.")
                    imgs_batch.append(torch.zeros((3, 299, 299), dtype=torch.float32))

            if imgs_batch:
                imgs_tensor = torch.stack(imgs_batch).to(device)
                if device.type == "cuda":
                    imgs_tensor = imgs_tensor.half()
                predictions_features.clear()
                with torch.no_grad():
                    _ = model(imgs_tensor)
                if predictions_features:
                    features.append(predictions_features[0])
    finally:
        handle.remove()

    if features:
        return np.vstack(features)
    return np.zeros((len(images_rgb), 1000), dtype=np.float32)


def extract_l2_features(
    model: torch.nn.Module,
    image_paths: Sequence[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Extract DF2 by computing gradient saliency per image."""
    del batch_size
    if model is None:
        print("[ERROR] Model is None, cannot extract L2 features")
        return np.zeros((len(image_paths), 1000), dtype=np.float32)

    model.eval()
    transform_eval = get_transforms(train=False)
    device = get_device()
    features = []
    hook_outputs = []

    def hook_fn(module, inputs, output):
        hook_outputs.append(output.detach().cpu().numpy())

    handle = model.classif.register_forward_hook(hook_fn)
    try:
        for path in tqdm(image_paths, desc="  L2 gradient saliency"):
            try:
                with Image.open(path) as img:
                    img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
                img_tensor = transform_eval(Image.fromarray(img_rgb))

                sal = gradient_saliency_map(model, img_tensor, device)
                sal_rgb = (np.stack([sal, sal, sal], axis=2) * 255).astype(np.uint8)
                sal_tensor = transform_eval(Image.fromarray(sal_rgb))

                hook_outputs.clear()
                with torch.no_grad():
                    sal_input = sal_tensor.unsqueeze(0).to(device)
                    if device.type == "cuda":
                        sal_input = sal_input.half()
                    _ = model(sal_input)

                if hook_outputs:
                    features.append(hook_outputs[0])
                else:
                    features.append(np.zeros((1, 1000), dtype=np.float32))
            except Exception as exc:
                print(f"[WARN] L2 failed for '{path}': {exc}")
                features.append(np.zeros((1, 1000), dtype=np.float32))
    finally:
        handle.remove()

    return np.vstack(features) if features else np.zeros((len(image_paths), 1000), dtype=np.float32)


def extract_l1_l2_batched(
    model: torch.nn.Module,
    image_paths: Sequence[str],
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract DF1 and DF2 in one forward pass per batch using original + saliency pairs."""
    if model is None:
        print("[ERROR] Model is None, cannot extract L1/L2 features")
        zeros = np.zeros((len(image_paths), 1000), dtype=np.float32)
        return zeros, zeros

    model.eval()
    transform = get_transforms(train=False)
    device = get_device()
    all_df1, all_df2 = [], []
    hook_outputs = []

    def hook_fn(module, inputs, output):
        hook_outputs.append(output.detach().cpu().numpy())

    handle = model.classif.register_forward_hook(hook_fn)
    try:
        for i in tqdm(range(0, len(image_paths), batch_size), desc="  L1+L2 batched"):
            batch_paths = image_paths[i : i + batch_size]
            originals, saliencies = [], []

            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
                    saliency_output = _saliency_attention.process(img_rgb)
                    orig_tensor = transform(Image.fromarray(saliency_output.enhanced_rgb))
                    sal = saliency_output.saliency_map
                    sal_rgb = (np.stack([sal, sal, sal], axis=2) * 255).astype(np.uint8)
                    sal_tensor = transform(Image.fromarray(sal_rgb))
                except Exception as exc:
                    print(f"[WARN] Failed to prepare '{path}': {exc}. Using zero tensors.")
                    orig_tensor = torch.zeros((3, 299, 299), dtype=torch.float32)
                    sal_tensor = torch.zeros((3, 299, 299), dtype=torch.float32)

                originals.append(orig_tensor)
                saliencies.append(sal_tensor)

            combined = torch.stack(originals + saliencies).to(device)
            if device.type == "cuda":
                combined = combined.half()

            hook_outputs.clear()
            with torch.no_grad():
                _ = model(combined)

            if hook_outputs:
                feats = hook_outputs[0]
                n = len(batch_paths)
                all_df1.append(feats[:n])
                all_df2.append(feats[n:])
            else:
                n = len(batch_paths)
                all_df1.append(np.zeros((n, 1000), dtype=np.float32))
                all_df2.append(np.zeros((n, 1000), dtype=np.float32))
    finally:
        handle.remove()

    df1 = np.vstack(all_df1) if all_df1 else np.zeros((len(image_paths), 1000), dtype=np.float32)
    df2 = np.vstack(all_df2) if all_df2 else np.zeros((len(image_paths), 1000), dtype=np.float32)
    return df1, df2


def _glcm_from_gray_array(arr: np.ndarray) -> np.ndarray:
    arr = (arr / 32).astype(np.uint8)
    arr = np.clip(arr, 0, 7)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    out: List[float] = []

    for angle in angles:
        glcm = graycomatrix(arr, distances=[1], angles=[angle], levels=8, symmetric=True, normed=True)
        out.append(float(graycoprops(glcm, "contrast")[0, 0]))
        out.append(float(graycoprops(glcm, "correlation")[0, 0]))
        out.append(float(graycoprops(glcm, "energy")[0, 0]))
        out.append(float(graycoprops(glcm, "homogeneity")[0, 0]))
        pmat = glcm[:, :, 0, 0]
        p_safe = np.where(pmat > 0, pmat, 1e-10)
        out.append(float(-np.sum(pmat * np.log2(p_safe))))

    return np.asarray(out, dtype=np.float32)


def extract_glcm_features_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Extract GLCM descriptors from an in-memory RGB image."""
    try:
        img = Image.fromarray(np.asarray(img_rgb, dtype=np.uint8)).convert("L")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr = np.asarray(img)
    except Exception as exc:
        print(f"[WARN] Could not prepare in-memory image for GLCM: {exc}")
        return np.zeros(20, dtype=np.float32)

    return _glcm_from_gray_array(arr)


def extract_glcm_features(image_path: str) -> np.ndarray:
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            arr = np.asarray(img)
    except UnidentifiedImageError as exc:
        print(f"[WARN] Corrupt image skipped in GLCM: '{image_path}'. Reason: {exc}")
        return np.zeros(20, dtype=np.float32)
    except OSError as exc:
        print(f"[WARN] Could not read image for GLCM '{image_path}': {exc}")
        return np.zeros(20, dtype=np.float32)

    return _glcm_from_gray_array(arr)


def extract_lbp_features(image_path: str) -> np.ndarray:
    """Extract the full LBP texture vector: histogram + statistics."""
    return _lbp_extractor.extract_from_path(image_path).feature_vector.astype(np.float32)


def extract_lbp_histogram(image_path: str) -> np.ndarray:
    """Extract legacy 10-bin LBP histogram for existing checkpoints."""
    return _lbp_extractor.extract_from_path(image_path).histogram.astype(np.float32)

def extract_all_glcm(image_paths: Sequence[str]) -> np.ndarray:
    hasher = hashlib.md5()
    for pth in image_paths:
        try:
            mtime = os.path.getmtime(pth)
            hasher.update(f"{pth}_{mtime}".encode("utf-8"))
        except OSError:
            hasher.update(pth.encode("utf-8"))

    cache_key = hasher.hexdigest()
    cache_file = npath(GLCM_CACHE_DIR, f"{cache_key}.npz")

    if os.path.exists(cache_file):
        print(f"[INFO] Loading GLCM features from cache: {cache_file}")
        data = np.load(cache_file)
        return data["arr_0"]

    print("[INFO] Extracting GLCM features in parallel...")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    workers = min(8, os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(executor.map(extract_glcm_features, image_paths), total=len(image_paths), desc="  GLCM features")
        )
    feats = np.asarray(results, dtype=np.float32)
    np.savez_compressed(cache_file, feats)
    return feats


def extract_all_lbp(image_paths: Sequence[str]) -> np.ndarray:
    hasher = hashlib.md5()
    hasher.update(f"lbp_dim_{LBP_FEATURE_DIM}_schema_v2".encode("utf-8"))
    for pth in image_paths:
        try:
            mtime = os.path.getmtime(pth)
            hasher.update(f"{pth}_{mtime}".encode("utf-8"))
        except OSError:
            hasher.update(pth.encode("utf-8"))

    cache_key = hasher.hexdigest()
    cache_file = npath(GLCM_CACHE_DIR, f"lbp_{cache_key}.npz")

    if os.path.exists(cache_file):
        print(f"[INFO] Loading LBP features from cache: {cache_file}")
        data = np.load(cache_file)
        cached = data["arr_0"]
        if cached.ndim == 2 and cached.shape[1] == LBP_FEATURE_DIM:
            return cached
        print(
            f"[WARN] Ignoring stale LBP cache '{cache_file}': "
            f"found width {cached.shape[1] if cached.ndim == 2 else cached.shape}, "
            f"expected {LBP_FEATURE_DIM}."
        )

    print("[INFO] Extracting LBP features in parallel...")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    workers = min(8, os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(executor.map(extract_lbp_features, image_paths), total=len(image_paths), desc="  LBP features")
        )
    feats = np.asarray(results, dtype=np.float32)
    np.savez_compressed(cache_file, feats)
    return feats


def _infer_label_map(objective: str, split_dir: str) -> Dict[str, str]:
    folder_names = [d for d in _safe_listdir(split_dir) if os.path.isdir(npath(split_dir, d))]
    lower = {d.lower() for d in folder_names}

    if objective == "informativeness" and {"informative", "not_informative"}.issubset(lower):
        return {name: name.lower() for name in folder_names}
    if objective == "damage" and {"severe", "mild", "little_or_none"}.issubset(lower):
        return {name: name.lower() for name in folder_names}
    return DEFAULT_LABEL_MAPS[objective]


def load_split(split: str, objective: str) -> Tuple[List[str], List[str]]:
    split_dir = npath(SPLIT_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"[ERROR] Split folder not found: {split_dir}")
        return [], []

    label_map = _infer_label_map(objective, split_dir)
    paths: List[str] = []
    labels: List[str] = []

    for folder in sorted(_safe_listdir(split_dir)):
        folder_dir = npath(split_dir, folder)
        if not os.path.isdir(folder_dir):
            continue

        mapped = label_map.get(folder)
        if mapped is None:
            mapped = label_map.get(folder.lower())
        if mapped is None:
            print(f"[WARN] Folder '{folder}' not mapped for objective '{objective}', skipping.")
            continue

        for fname in _safe_listdir(folder_dir):
            if _is_image(fname):
                paths.append(npath(folder_dir, fname))
                labels.append(mapped)

    return paths, labels


def _save_metrics_json(objective: str, metrics: Dict[str, float]) -> str:
    out_path = npath(RESULT_DIR, f"metrics_{objective}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fobj:
            json.dump(metrics, fobj, indent=2)
    except OSError as exc:
        print(f"[WARN] Could not write metrics json '{out_path}': {exc}")
    return out_path


def _save_confusion_plot(cm: np.ndarray, labels: Sequence[str], objective: str) -> Optional[str]:
    if plt is None:
        return None
    out_path = npath(RESULT_DIR, f"confusion_matrix_{objective}.png")
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {objective}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except OSError as exc:
        print(f"[WARN] Could not save confusion matrix image '{out_path}': {exc}")
        return None


def _feature_cache_path(objective: str, split: str) -> str:
    return npath(FEATURE_CACHE_DIR, f"{objective}_{split}_{FEATURE_PIPELINE}_features.npz")


def save_feature_cache(
    objective: str,
    split: str,
    image_paths: Sequence[str],
    labels: Sequence[str],
    df1: np.ndarray,
    df2: np.ndarray,
    glcm: np.ndarray,
    lbp: Optional[np.ndarray],
    fused: np.ndarray,
) -> str:
    out_path = _feature_cache_path(objective, split)
    np.savez_compressed(
        out_path,
        image_paths=np.asarray(image_paths, dtype=object),
        labels=np.asarray(labels, dtype=object),
        df1=df1.astype(np.float32),
        df2=df2.astype(np.float32),
        glcm=glcm.astype(np.float32),
        lbp=np.asarray(lbp, dtype=np.float32) if lbp is not None else np.zeros((len(image_paths), 0), dtype=np.float32),
        fused=fused.astype(np.float32),
    )
    return out_path


def load_cached_feature_matrix(
    objective: str,
    split: str,
    feature_set: str = "fused",
    feature_pipeline: str = FEATURE_PIPELINE,
) -> Optional[np.ndarray]:
    cache_path = npath(FEATURE_CACHE_DIR, f"{objective}_{split}_{feature_pipeline}_features.npz")
    if not os.path.exists(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=True)
    if feature_set not in data:
        return None
    return data[feature_set]


def _load_shared_feature_models() -> Tuple[torch.nn.Module, torch.nn.Module]:
    if "feature_extractor" not in _shared_feature_models:
        model = load_frozen_model(num_classes=1000)
        if model is None:
            raise RuntimeError("Could not initialize shared feature extractors")
        _shared_feature_models["feature_extractor"] = model

    model = _shared_feature_models["feature_extractor"]
    return model, model


@lru_cache(maxsize=len(OBJECTIVES))
def _load_rf_bundle(objective: str):
    candidates = [
        (npath(MODEL_DIR, f"rf_{objective}_{FEATURE_PIPELINE}.joblib"),
         npath(MODEL_DIR, f"le_{objective}_{FEATURE_PIPELINE}.joblib"),
         npath(MODEL_DIR, f"scaler_{objective}_{FEATURE_PIPELINE}.joblib"),
         FEATURE_PIPELINE),
        (npath(MODEL_DIR, f"rf_{objective}_{LEGACY_FEATURE_PIPELINE}.joblib"),
         npath(MODEL_DIR, f"le_{objective}_{LEGACY_FEATURE_PIPELINE}.joblib"),
         npath(MODEL_DIR, f"scaler_{objective}_{LEGACY_FEATURE_PIPELINE}.joblib"),
         LEGACY_FEATURE_PIPELINE),
        (npath(MODEL_DIR, f"rf_{objective}.joblib"),
         npath(MODEL_DIR, f"le_{objective}.joblib"),
         npath(MODEL_DIR, f"scaler_{objective}.joblib"),
         "df1_df2_glcm"),
    ]
    for clf_path, le_path, scaler_path, pipeline in candidates:
        if os.path.exists(clf_path) and os.path.exists(le_path) and os.path.exists(scaler_path):
            clf = joblib.load(clf_path)
            le = joblib.load(le_path)
            scaler = joblib.load(scaler_path)
            expected_dim = getattr(scaler, "n_features_in_", None)
            spec = resolve_artifact_spec(pipeline, expected_dim)
            if spec.name != pipeline:
                print(
                    f"[WARN] Artifact '{os.path.basename(scaler_path)}' is named '{pipeline}' "
                    f"but scaler expects {expected_dim} features. Using '{spec.name}' contract."
                )
            return clf, le, scaler, spec
    raise FileNotFoundError(f"Missing inference artifacts for objective '{objective}'")

def predict_image(image_path: str, objective: str) -> Dict[str, object]:
    """Run the paper-aligned fused inference path for one image."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective '{objective}'")

    clf, le, scaler, feature_spec = _load_rf_bundle(objective)
    model_l1, model_l2 = _load_shared_feature_models()

    df1 = extract_features_predictions_layer(model_l1, [image_path], batch_size=1, saliency_input=False)
    df2 = extract_features_predictions_layer(model_l2, [image_path], batch_size=1, saliency_input=True)
    glcm = np.asarray([extract_glcm_features(image_path)], dtype=np.float32)
    if feature_spec.name == FEATURE_PIPELINE:
        lbp = np.asarray([extract_lbp_features(image_path)], dtype=np.float32)
        fused = np.hstack([df1, df2, glcm, lbp])
    elif feature_spec.name == LEGACY_FEATURE_PIPELINE:
        lbp = np.asarray([extract_lbp_histogram(image_path)], dtype=np.float32)
        fused = np.hstack([df1, df2, glcm, lbp])
    else:
        fused = np.hstack([df1, df2, glcm])
    fused_scaled = scaler.transform(fused)

    pred_idx = int(clf.predict(fused_scaled)[0])
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = None
    probabilities = None
    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(fused_scaled)[0]
        confidence = float(np.max(probabilities))

    result = {
        "objective": objective,
        "prediction": pred_label,
        "feature_dim": int(fused.shape[1]),
        "feature_pipeline": feature_spec.name,
        "feature_metadata": feature_spec.metadata(),
    }
    if confidence is not None:
        result["confidence"] = confidence
        result["probabilities"] = {
            cls_name: float(prob) for cls_name, prob in zip(le.classes_, probabilities)
        }
    return result


def train_and_evaluate(objective: str, torch_cfg: Optional[TorchTrainingConfig] = None) -> Optional[Dict[str, float]]:
    if objective not in OBJECTIVES:
        print(f"[ERROR] Unknown objective '{objective}'. Valid: {OBJECTIVES}")
        return None

    print("\n" + "=" * 70)
    print(f"STEP 3 - M1 Module (Paper: Gupta & Roy 2025) - Objective: {objective}")
    print("=" * 70)

    train_paths, train_labels = load_split("train", objective)
    test_paths, test_labels = load_split("test", objective)
    if not train_paths or not test_paths:
        print("[ERROR] Missing train/test image paths. Run step2_preprocess.py first.")
        return None

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("DisasterRes-Net")

    with mlflow.start_run(run_name=f"rf_{objective}"):
        mlflow.set_tags(
            {
                "framework": "pytorch+sklearn",
                "objective": objective,
                "backbone": "timm/inception_resnet_v2",
                "feature_pipeline": FEATURE_PIPELINE,
            }
        )

        if torch_cfg is not None:
            aux_metrics = train_pytorch_backbone(objective, train_paths, y_train, le, torch_cfg)
            if aux_metrics:
                print(f"[INFO] Optional PyTorch classifier metrics: {aux_metrics}")

        print("\n[STEP] Loading frozen pretrained Inception-ResNet-V2 for L1 (original image)")
        model_l1 = load_frozen_model(num_classes=1000)
        if model_l1 is None:
            return None

        print("\n[STEP] Extracting L1 + L2 features in single batched pass")
        df1_train, df2_train = extract_l1_l2_batched(model_l1, train_paths)
        df1_test, df2_test = extract_l1_l2_batched(model_l1, test_paths)
        print(f"  DF1 train shape: {df1_train.shape}, DF1 test shape: {df1_test.shape}")
        print(f"  DF2 train shape: {df2_train.shape}, DF2 test shape: {df2_test.shape}")

        del model_l1
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[STEP] Extracting L3 handcrafted features (GLCM) - HF (20-dim)")
        glcm_train = extract_all_glcm(train_paths)
        glcm_test = extract_all_glcm(test_paths)
        print(f"  GLCM train shape: {glcm_train.shape}, GLCM test shape: {glcm_test.shape}")

        print("[STEP] Extracting L4 handcrafted features (LBP) - LT (10-dim)")
        lbp_train = extract_all_lbp(train_paths)
        lbp_test = extract_all_lbp(test_paths)
        print(f"  LBP train shape: {lbp_train.shape}, LBP test shape: {lbp_test.shape}")

        print("\n[STEP] Fusing features: [DF1(1000) | DF2(1000) | HF(20) | LT(10)] = 2030-dim")
        x_train_raw = np.hstack([df1_train, df2_train, glcm_train, lbp_train])
        x_test_raw = np.hstack([df1_test, df2_test, glcm_test, lbp_test])
        print(f"  Fused train shape: {x_train_raw.shape}")
        print(f"  Fused test shape:  {x_test_raw.shape}")

        train_cache_path = save_feature_cache(
            objective,
            "train",
            train_paths,
            train_labels,
            df1_train,
            df2_train,
            glcm_train,
            lbp_train,
            x_train_raw,
        )
        test_cache_path = save_feature_cache(
            objective,
            "test",
            test_paths,
            test_labels,
            df1_test,
            df2_test,
            glcm_test,
            lbp_test,
            x_test_raw,
        )

        print("[STEP] Normalizing fused features")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)

        print("\n[STEP] Training Random Forest classifier (paper-aligned)")
        clf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        print("[INFO] Running Stratified K-Fold Cross Validation (5 folds)...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=skf, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        print(f"[RESULT] CV Accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")

        try:
            clf.fit(x_train, y_train)
        except Exception as exc:
            print(f"[ERROR] Random Forest training failed: {exc}")
            return None

        y_pred = clf.predict(x_test)
        acc = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n[RESULT] Test accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        print("[REPORT]")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        cm_path = _save_confusion_plot(cm, le.classes_, objective)

        model_path = npath(MODEL_DIR, f"rf_{objective}_{FEATURE_PIPELINE}.joblib")
        encoder_path = npath(MODEL_DIR, f"le_{objective}_{FEATURE_PIPELINE}.joblib")
        scaler_path = npath(MODEL_DIR, f"scaler_{objective}_{FEATURE_PIPELINE}.joblib")
        try:
            joblib.dump(clf, model_path)
            joblib.dump(le, encoder_path)
            joblib.dump(scaler, scaler_path)
            if objective == "informativeness":
                compat_encoder_path = npath(MODEL_DIR, "le_disaster_type.joblib")
                joblib.dump(le, compat_encoder_path)
        except OSError as exc:
            print(f"[ERROR] Failed saving model artifacts: {exc}")
            return None

        df1_names = [f"df1_{i + 1}" for i in range(1000)]
        df2_names = [f"df2_{i + 1}" for i in range(1000)]
        glcm_names = []
        for ang in ["0", "45", "90", "135"]:
            for prop in ["contrast", "correlation", "energy", "homogeneity", "entropy"]:
                glcm_names.append(f"glcm_{prop}_{ang}")
        feature_spec = get_pipeline_spec(FEATURE_PIPELINE)
        feature_names = feature_spec.feature_names
        validate_feature_importance_length(clf.feature_importances_, feature_spec)

        importances_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
        importances_df = importances_df.sort_values(by="importance", ascending=False)
        importances_path = npath(RESULT_DIR, f"feature_importances_{objective}.csv")
        importances_df.to_csv(importances_path, index=False)

        metrics = {
            "objective": objective,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "train_images": float(len(train_paths)),
            "test_images": float(len(test_paths)),
            "feature_dim": float(x_train.shape[1]),
            "classifier": "RandomForest",
            "architecture": "M1_Paper",
            "backbone_framework": "PyTorch",
            "backbone_model": "timm/inception_resnet_v2",
            "l1_dim": 1000.0,
            "l2_dim": 1000.0,
            "l3_dim": 20.0,
            "l4_dim": float(LBP_FEATURE_DIM),
        }
        metrics_path = _save_metrics_json(objective, metrics)

        mlflow.log_param("objective", objective)
        mlflow.log_params(
            {
                "n_estimators": 300,
                "class_weight": "balanced",
                "max_features": "sqrt",
                "random_state": 42,
                "classifier": "RandomForestClassifier",
                "architecture": "M1-Paper",
                "fusion_dim": 2036,
                "pca_applied": False,
                "backbone_framework": "PyTorch",
                "backbone_library": "timm",
                "backbone_model": "inception_resnet_v2",
                "normalization": "imagenet",
                "saliency_method": "saliency_attention_weighted_cnn_input",
                "feature_pipeline": FEATURE_PIPELINE,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
                "train_images": float(len(train_paths)),
                "test_images": float(len(test_paths)),
                "feature_dim": float(x_train.shape[1]),
            }
        )
        mlflow.sklearn.log_model(clf, f"rf_model_{objective}")
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(train_cache_path)
        mlflow.log_artifact(test_cache_path)
        if cm_path:
            mlflow.log_artifact(cm_path)
        mlflow.log_artifact(importances_path)
        mlflow.log_artifact(metrics_path)

        inference_spec = {
            "objective": objective,
            "classifier_artifact": os.path.basename(model_path),
            "encoder_artifact": os.path.basename(encoder_path),
            "scaler_artifact": os.path.basename(scaler_path),
            "feature_cache_train": os.path.basename(train_cache_path),
            "feature_cache_test": os.path.basename(test_cache_path),
            "deep_feature_source": "inception_resnet_v2 logits",
            "feature_pipeline": FEATURE_PIPELINE,
        }
        spec_path = npath(RESULT_DIR, f"inference_spec_{objective}.json")
        with open(spec_path, "w", encoding="utf-8") as fobj:
            json.dump(inference_spec, fobj, indent=2)
        mlflow.log_artifact(spec_path)

        print(f"\n[INFO] Model saved: {model_path}")
        print(f"[INFO] Label encoder saved: {encoder_path}")
        print(f"[INFO] Scaler saved: {scaler_path}")
        print(f"[INFO] Feature caches saved: {train_cache_path}, {test_cache_path}")
        if cm_path:
            print(f"[INFO] Confusion matrix saved: {cm_path}")
        print(f"[INFO] Metrics saved: {metrics_path}")
        return metrics


def main() -> int:
    if torch is None or timm is None:
        return 1

    torch_cfg = TorchTrainingConfig()
    all_ok = True
    for objective in OBJECTIVES:
        result = train_and_evaluate(objective, torch_cfg=torch_cfg)
        if result is None:
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())





