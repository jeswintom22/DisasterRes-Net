"""Step 3: extract features, train XGBoost, evaluate, and save models."""

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib
import concurrent.futures

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from PIL import Image, UnidentifiedImageError
from xgboost import XGBClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    print(f"[WARN] matplotlib import failed: {exc}. Confusion image generation will be disabled.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import timm
except Exception as exc:
    torch = None
    timm = None
    print(f"[ERROR] PyTorch / timm import failed: {exc}")
    print("[ERROR] Install torch, torchvision, and timm for Python 3.10 (Windows), then retry.")

_device = None
def get_device():
    global _device
    if _device is None and torch is not None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _device.type == "cuda":
            print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] No GPU found — running on CPU (slower).")
    return _device


def npath(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


SPLIT_DIR = npath("split_dataset")
MODEL_DIR = npath("saved_models")
RESULT_DIR = npath("results")
GLCM_CACHE_DIR = npath(".cache")
IMG_SIZE = (299, 299)
BATCH_SIZE = 64

OBJECTIVES = ("informativeness", "damage")
IMG_EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GLCM_CACHE_DIR, exist_ok=True)


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


def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError as exc:
        print(f"[ERROR] Could not list directory '{path}': {exc}")
        return []


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)


class DisasterDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception:
            return torch.zeros((3, 299, 299), dtype=torch.float32), label


def get_transforms(train: bool = False):
    if train:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def finetune_extractor(train_paths: Sequence[str], y_train: np.ndarray, num_classes: int, objective: str):
    if timm is None or torch is None:
        return None
    try:
        model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=num_classes)
        
        # Freeze all parameters first
        named_params = list(model.named_parameters())
        for name, param in named_params:
            param.requires_grad = False
            
        # Unfreeze the last 100 parameter tensors
        for name, param in named_params[-100:]:
            param.requires_grad = True

        device = get_device()
        model = model.to(device)
        
        # Partition parameters for differential learning rates
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'classif' in name or 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
                    
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 1e-4}
        ])
        criterion = nn.CrossEntropyLoss()
        
        transform = get_transforms(train=True)
        dataset = DisasterDataset(train_paths, y_train, transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        print(f"[INFO] Fine-tuning InceptionResNetV2 backbone for 5 epochs...")
        model.train()
        for epoch in range(5):
            running_loss = 0.0
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
            epoch_loss = running_loss / len(dataset)
            print(f"Epoch {epoch+1}/5 Loss: {epoch_loss:.4f}")
            
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = npath(MODEL_DIR, f"model_{objective}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Saved fine-tuned PyTorch model to {model_path}")
        
        if objective == "informativeness":
            compat_model_path = npath(MODEL_DIR, "model_disaster_type.pth")
            torch.save(model.state_dict(), compat_model_path)
            print(f"[INFO] Saved compatibility duplicate model to {compat_model_path}")

        model.reset_classifier(0)
        return model
    except Exception as exc:
        print(f"[ERROR] Failed to fine-tune PyTorch model: {exc}")
        return None


def extract_deep_features(model, image_paths: Sequence[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    model.eval()
    transform = get_transforms(train=False)
    dataset = DisasterDataset(image_paths, [0]*len(image_paths), transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = get_device()
    features = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="  Deep features"):
            imgs = imgs.to(device)
            try:
                preds = model(imgs)
                features.append(preds.cpu().numpy())
            except Exception as exc:
                print(f"[ERROR] Deep extraction failed: {exc}")
                features.append(np.zeros((imgs.size(0), 1536), dtype=np.float32))
                
    if features:
        return np.vstack(features)
    return np.array([])


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


def extract_all_glcm(image_paths: Sequence[str]) -> np.ndarray:
    hasher = hashlib.md5()
    for pth in image_paths:
        try:
            mtime = os.path.getmtime(pth)
            hasher.update(f"{pth}_{mtime}".encode('utf-8'))
        except OSError:
            hasher.update(pth.encode('utf-8'))
    
    cache_key = hasher.hexdigest()
    cache_file = npath(GLCM_CACHE_DIR, f"{cache_key}.npz")
    
    if os.path.exists(cache_file):
        print(f"[INFO] Loading GLCM features from cache: {cache_file}")
        data = np.load(cache_file)
        return data['arr_0']
        
    print("[INFO] Extracting GLCM features in parallel...")
    feats = []
    
    # Prevent OpenBLAS from creating thread pools in every child process, 
    # which exhausts system RAM when multiplied by the number of workers.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Limit workers to avoid out-of-memory errors on laptops
    workers = min(8, os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(extract_glcm_features, image_paths), total=len(image_paths), desc="  GLCM features"))
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


def train_and_evaluate(objective: str) -> Optional[Dict[str, float]]:
    if objective not in OBJECTIVES:
        print(f"[ERROR] Unknown objective '{objective}'. Valid: {OBJECTIVES}")
        return None

    print("\n" + "=" * 70)
    print(f"STEP 3 - Objective: {objective}")
    print("=" * 70)

    train_paths, train_labels = load_split("train", objective)
    test_paths, test_labels = load_split("test", objective)

    if not train_paths or not test_paths:
        print("[ERROR] Missing train/test image paths. Run step2_preprocess.py first.")
        return None

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)

    extractor = finetune_extractor(train_paths, y_train, len(le.classes_), objective)
    if extractor is None:
        print("[ERROR] Feature extractor unavailable or fine-tuning failed.")
        return None

    deep_train = extract_deep_features(extractor, train_paths)
    deep_test = extract_deep_features(extractor, test_paths)
    
    # Free up GPU memory
    del extractor
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    glcm_train = extract_all_glcm(train_paths)
    glcm_test = extract_all_glcm(test_paths)

    if deep_train.shape[1] != 1536:
        print(f"[WARN] Deep feature dim is {deep_train.shape[1]}, expected 1536.")
    if glcm_train.shape[1] != 20:
        print(f"[WARN] GLCM feature dim is {glcm_train.shape[1]}, expected 20.")

    print("[INFO] Applying PCA to deep features (1536 -> 64)")
    pca = PCA(n_components=64, random_state=42)
    deep_train_pca = pca.fit_transform(deep_train)
    deep_test_pca = pca.transform(deep_test)

    x_train = np.hstack([deep_train_pca, glcm_train])
    x_test = np.hstack([deep_test_pca, glcm_test])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print(f"[INFO] Fused feature shape train: {x_train.shape} (normalized)")
    print(f"[INFO] Fused feature shape test : {x_test.shape} (normalized)")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("DisasterRes-Net")
    with mlflow.start_run(run_name=f"rf_{objective}"):
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric='mlogloss',
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )
        
        print("[INFO] Running Stratified K-Fold Cross Validation (5 folds)...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=skf, scoring='accuracy')
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        print(f"[RESULT] CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

        try:
            clf.fit(x_train, y_train)
        except Exception as exc:
            print(f"[ERROR] XGBoost training failed: {exc}")
            return None
    
        y_pred = clf.predict(x_test)
        acc = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
    
        print(f"[RESULT] Test accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        print("[REPORT]")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
        cm_path = _save_confusion_plot(cm, le.classes_, objective)
    
        model_path = npath(MODEL_DIR, f"rf_{objective}.joblib")
        encoder_path = npath(MODEL_DIR, f"le_{objective}.joblib")
        scaler_path = npath(MODEL_DIR, f"scaler_{objective}.joblib")
        pca_path = npath(MODEL_DIR, f"pca_{objective}.joblib")
        try:
            joblib.dump(clf, model_path)
            joblib.dump(le, encoder_path)
            if objective == "informativeness":
                compat_encoder_path = npath(MODEL_DIR, "le_disaster_type.joblib")
                joblib.dump(le, compat_encoder_path)
                print(f"[INFO] Saved compatibility duplicate label encoder to {compat_encoder_path}")
            joblib.dump(scaler, scaler_path)
            joblib.dump(pca, pca_path)
        except OSError as exc:
            print(f"[ERROR] Failed saving model artifacts: {exc}")
            return None

        pca_names = [f"deep_pca_{i+1}" for i in range(64)]
        glcm_names = []
        for ang in ["0", "45", "90", "135"]:
            for prop in ["contrast", "correlation", "energy", "homogeneity", "entropy"]:
                glcm_names.append(f"glcm_{prop}_{ang}")
        feature_names = pca_names + glcm_names
        
        importances_df = pd.DataFrame({
            "feature": feature_names,
            "importance": clf.feature_importances_
        }).sort_values(by="importance", ascending=False)
        
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
        }
        metrics_path = _save_metrics_json(objective, metrics)
        
        mlflow.log_param("objective", objective)
        mlflow.log_params({
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "random_state": 42,
            "pca_components": 32
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "train_images": float(len(train_paths)),
            "test_images": float(len(test_paths)),
            "feature_dim": float(x_train.shape[1])
        })
        mlflow.sklearn.log_model(clf, f"xgb_model_{objective}")
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(pca_path)
        if cm_path:
            mlflow.log_artifact(cm_path)
        mlflow.log_artifact(importances_path)
        mlflow.log_artifact(metrics_path)
    
        print(f"[INFO] Model saved: {model_path}")
        print(f"[INFO] Label encoder saved: {encoder_path}")
        print(f"[INFO] Scaler saved: {scaler_path}")
        print(f"[INFO] PCA saved: {pca_path}")
        print(f"[INFO] Feature importances saved: {importances_path}")
        if cm_path:
            print(f"[INFO] Confusion matrix saved: {cm_path}")
        print(f"[INFO] Metrics saved: {metrics_path}")
    
        return metrics


def main() -> int:
    all_ok = True
    for objective in OBJECTIVES:
        result = train_and_evaluate(objective)
        if result is None:
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
