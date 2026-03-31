"""
Step 3 — End-to-End PyTorch Deep Learning Classification.
Backend: PyTorch + timm (InceptionResNetV2).
This script directly fine-tunes the CNN for disaster type and damage level.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Tuple, Sequence

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms as T

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except Exception as _e:
    plt = None
    _PLT = False

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[INFO] CUDA GPU : {_gpu_name}  ({_vram_gb:.1f} GB VRAM)")
else:
    print("[WARN] No CUDA GPU — running on CPU. Training will be slow.")

def npath(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))

SPLIT_DIR  = npath("split_dataset")
MODEL_DIR  = npath("saved_models")
RESULT_DIR = npath("results")

IMG_SIZE   = (299, 299)
BATCH_SIZE = 32  # Reduced for training backprop
EPOCHS     = 10
NUM_WORKERS = min(4, os.cpu_count() or 1)

OBJECTIVES = ("disaster_type", "damage")
IMG_EXTS   = (".jpg", ".jpeg", ".png")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DISASTER_TYPE_FOLDERS = {
    "earthquake", "flood", "hurricane",
    "wildfire", "landslide", "not_disaster",
}

DAMAGE_LABEL_MAP: Dict[str, str] = {
    "earthquake":   "severe",
    "hurricane":    "severe",
    "landslide":    "severe",
    "flood":        "mild",
    "wildfire":     "mild",
    "not_disaster": "little_or_none",
}

def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError as exc:
        print(f"[ERROR] Cannot list '{path}': {exc}")
        return []

def _is_image(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)

def load_split(split: str, objective: str) -> Tuple[List[str], List[str]]:
    split_dir = npath(SPLIT_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"[ERROR] Split folder not found: {split_dir}")
        return [], []

    paths: List[str]  = []
    labels: List[str] = []

    for folder in sorted(_safe_listdir(split_dir)):
        folder_dir   = npath(split_dir, folder)
        folder_lower = folder.lower()
        if not os.path.isdir(folder_dir):
            continue

        if objective == "disaster_type":
            if folder_lower not in DISASTER_TYPE_FOLDERS:
                continue
            label = folder_lower
        elif objective == "damage":
            label = DAMAGE_LABEL_MAP.get(folder_lower)
            if label is None:
                continue
        else:
            return [], []

        for fname in _safe_listdir(folder_dir):
            if _is_image(fname):
                paths.append(npath(folder_dir, fname))
                labels.append(label)

    print(f"[INFO] {split:5s} | {objective:15s} | {len(paths)} images")
    return paths, labels

class DisasterDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback to a black image if corrupted
            img = Image.new("RGB", IMG_SIZE, (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def get_transforms(is_train: bool):
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return T.Compose([
            T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def get_class_weights(y: np.ndarray, num_classes: int) -> torch.FloatTensor:
    classes = np.arange(num_classes)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return torch.FloatTensor(w).to(_DEVICE)

def build_model(num_classes: int) -> nn.Module:
    model = timm.create_model("inception_resnet_v2", pretrained=True, num_classes=num_classes)
    return model.to(_DEVICE)

def train_epoch(model, dataloader, criterion, optimizer, scaler) -> float:
    model.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(_DEVICE), targets.to(_DEVICE)

        optimizer.zero_grad(set_to_none=True)
        # Mixed precision training
        with torch.amp.autocast(_DEVICE.type):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(_DEVICE)
            with torch.amp.autocast(_DEVICE.type):
                outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)

def train_and_evaluate(objective: str) -> Optional[Dict]:
    print("\n" + "=" * 72)
    print(f"  OBJECTIVE : {objective}")
    print("=" * 72)

    train_paths, train_labels = load_split("train", objective)
    test_paths,  test_labels  = load_split("test",  objective)
    if not train_paths or not test_paths:
        print("[ERROR] Empty split — run preprocessing first.")
        return None

    le = LabelEncoder()
    le.fit(train_labels + test_labels)
    y_train = le.transform(train_labels)
    y_test  = le.transform(test_labels)
    num_classes = len(le.classes_)
    print(f"[INFO] Classes : {list(le.classes_)}")

    train_ds = DisasterDataset(train_paths, y_train, get_transforms(is_train=True))
    test_ds  = DisasterDataset(test_paths,  y_test,  get_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    class_weights = get_class_weights(y_train, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = build_model(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(_DEVICE.type)

    best_f1 = 0.0
    model_save_path = npath(MODEL_DIR, f"model_{objective}.pth")
    le_save_path = npath(MODEL_DIR, f"le_{objective}.joblib")

    print(f"\n[TRAIN] Beginning fine-tuning for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        preds, targets = evaluate(model, test_loader)
        
        val_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0))
        val_acc = float(accuracy_score(targets, preds))
        
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Train Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model! (F1: {best_f1:.4f})")

    print("\n[EVAL] Final evaluation with best model...")
    model.load_state_dict(torch.load(model_save_path, map_location=_DEVICE, weights_only=True))
    preds, targets = evaluate(model, test_loader)

    acc       = float(accuracy_score(targets, preds))
    precision = float(precision_score(targets, preds, average="weighted", zero_division=0))
    recall    = float(recall_score(targets, preds, average="weighted", zero_division=0))
    f1        = float(f1_score(targets, preds, average="weighted", zero_division=0))

    print(f"\n[RESULT] acc={acc:.4f}  prec={precision:.4f}  rec={recall:.4f}  f1={f1:.4f}")
    print(classification_report(targets, preds, target_names=le.classes_, zero_division=0))

    joblib.dump(le, le_save_path)
    
    if _PLT:
        cm = confusion_matrix(targets, preds)
        path = npath(RESULT_DIR, f"confusion_matrix_{objective}.png")
        fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
        ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion matrix — {objective}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

    metrics = {
        "objective":    objective,
        "accuracy":     acc,
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
        "train_images": len(train_paths),
        "test_images":  len(test_paths),
    }
    
    metrics_path = npath(RESULT_DIR, f"metrics_{objective}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def predict_single_image(image_path: str) -> Dict[str, str]:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return {"error": f"Could not load {image_path}: {exc}"}

    transform = get_transforms(is_train=False)
    t = transform(img).unsqueeze(0).to(_DEVICE)

    results: Dict[str, str] = {}
    for obj in OBJECTIVES:
        le_path = npath(MODEL_DIR, f"le_{obj}.joblib")
        model_path = npath(MODEL_DIR, f"model_{obj}.pth")
        
        if not os.path.exists(le_path) or not os.path.exists(model_path):
            results[obj] = "model_not_trained"
            continue
            
        le = joblib.load(le_path)
        model = build_model(len(le.classes_))
        model.load_state_dict(torch.load(model_path, map_location=_DEVICE, weights_only=True))
        model.eval()

        with torch.no_grad():
            with torch.amp.autocast(_DEVICE.type):
                outputs = model(t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        pred_label = le.inverse_transform([pred_idx.item()])[0]
        results[obj] = pred_label
        results[f"{obj}_confidence"] = f"{conf.item():.3f}"

    return results

def main() -> int:
    summary: List[Dict] = []
    all_ok = True

    for objective in OBJECTIVES:
        result = train_and_evaluate(objective)
        if result is None:
            all_ok = False
        else:
            summary.append(result)

    if summary:
        print("\n" + "=" * 72)
        print(f"  {'Objective':<20} {'Acc':>7} {'F1':>7}")
        print("-" * 64)
        for m in summary:
            print(f"  {m['objective']:<20} {m['accuracy']:>7.4f} {m['f1']:>7.4f}")
        print("=" * 72)

    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())