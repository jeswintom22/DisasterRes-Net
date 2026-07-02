"""Step 3: extract features using M1 architecture (paper: Gupta & Roy 2025).

Architecture:
  L1: IRv2 on original image → DF1 (1000-dim)
  L2: IRv2 on saliency map  → DF2 (1000-dim)
  L3: GLCM handcrafted      → HF  (20-dim)
  Fusion: [DF1 | DF2 | HF] = 2020-dim (no PCA)
  Classifier: Random Forest (not XGBoost)
"""

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
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


def rgb_to_lms(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to LMS color space (paper step 1 of SUN model).
    
    Args:
        img_rgb: RGB image (H, W, 3) with values in [0, 255]
    
    Returns:
        LMS image (H, W, 3)
    """
    # Normalize to [0, 1]
    img = img_rgb.astype(np.float32) / 255.0
    
    # Standard RGB to LMS transformation matrix
    # (This is an approximation; exact matrix from Gupta & Roy may vary slightly)
    rgb_to_lms_matrix = np.array([
        [0.3, 0.622, 0.078],
        [0.23, 0.692, 0.078],
        [0.24342268924547819, 0.20476744424496821, 0.55314986650955360]
    ])
    
    h, w = img_rgb.shape[:2]
    img_flat = img.reshape(-1, 3)
    lms_flat = img_flat @ rgb_to_lms_matrix.T
    lms = lms_flat.reshape(h, w, 3)
    
    # Clip to avoid numerical issues
    return np.clip(lms, 0, 1)


def simple_saliency_map(img_rgb: np.ndarray) -> np.ndarray:
    """Generate approximate saliency map using Laplacian contrast.
    
    This is a practical approximation of the SUN model. The exact SUN model
    requires 361 pre-trained ICA filters from Kanan & Cottrell (2010), which
    we document as a TODO.
    
    Args:
        img_rgb: RGB image (H, W, 3) with values in [0, 255]
    
    Returns:
        Saliency map (H, W) with values in [0, 1]
    """
    try:
        # Convert to grayscale
        img_gray = Image.fromarray(img_rgb.astype(np.uint8)).convert('L')
        img_gray = np.asarray(img_gray, dtype=np.float32) / 255.0
        
        # Apply Laplacian for edge detection (high contrast → high saliency)
        from scipy.ndimage import laplace
        sal = np.abs(laplace(img_gray))
        
        # Normalize to [0, 1]
        sal_min, sal_max = sal.min(), sal.max()
        if sal_max > sal_min:
            sal = (sal - sal_min) / (sal_max - sal_min)
        else:
            sal = np.zeros_like(sal)
        
        return sal
    except Exception as exc:
        print(f"[WARN] Saliency extraction failed: {exc}. Returning uniform saliency.")
        return np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32) * 0.5


def saliency_to_rgb(saliency_map: np.ndarray) -> np.ndarray:
    """Convert 2D saliency map to 3-channel RGB for model input.
    
    Args:
        saliency_map: (H, W) saliency values in [0, 1]
    
    Returns:
        (H, W, 3) RGB image in [0, 255]
    """
    # Replicate saliency across 3 channels
    rgb_sal = np.stack([saliency_map, saliency_map, saliency_map], axis=2)
    return (rgb_sal * 255).astype(np.uint8)


def load_frozen_model(num_classes: int = 1000) -> Optional[torch.nn.Module]:
    """Load frozen pretrained Inception-ResNet-V2 (paper's L1 & L2 source).
    
    The paper uses pretrained IRv2 without fine-tuning. This extracts features
    from the predictions layer (1000-dim), not global average pooling.
    
    Args:
        num_classes: Number of ImageNet classes (1000)
    
    Returns:
        Model with forward hooks registered to capture predictions layer
    """
    if timm is None or torch is None:
        print("[ERROR] PyTorch / timm not available")
        return None
    
    try:
        model = timm.create_model(
            'inception_resnet_v2',
            pretrained=True,
            num_classes=num_classes
        )
        
        # Freeze all parameters (no fine-tuning per paper)
        for param in model.parameters():
            param.requires_grad = False
        
        device = get_device()
        model = model.to(device)
        model.eval()
        
        print("[INFO] Loaded frozen pretrained Inception-ResNet-V2")
        return model
    except Exception as exc:
        print(f"[ERROR] Failed to load frozen model: {exc}")
        return None


def extract_features_predictions_layer(model, image_paths: Sequence[str], 
                                       batch_size: int = BATCH_SIZE,
                                       saliency_input: bool = False) -> np.ndarray:
    """Extract 1000-dim features from IRv2 predictions layer (paper's L1/L2).
    
    This replaces the old extract_deep_features which used GAP (1536-dim).
    The paper uses the predictions layer (1000-dim before softmax).
    
    Args:
        model: Frozen pretrained Inception-ResNet-V2
        image_paths: List of image file paths
        batch_size: Batch size for extraction
        saliency_input: If True, generate saliency map for each image first
    
    Returns:
        Feature matrix (N, 1000)
    """
    if model is None:
        print("[ERROR] Model is None, cannot extract features")
        return np.zeros((len(image_paths), 1000), dtype=np.float32)
    
    model.eval()
    transform = get_transforms(train=False)
    device = get_device()
    
    # Hook to capture predictions layer output (1000-dim)
    predictions_features = []
    
    def hook_fn(module, input, output):
        predictions_features.append(output.detach().cpu().numpy())
    
    # Register hook on the classification layer
    handle = model.classif.register_forward_hook(hook_fn)
    
    features = []
    try:
        for i in tqdm(range(0, len(image_paths), batch_size), desc="  L1/L2 features"):
            batch_paths = image_paths[i:i+batch_size]
            imgs_batch = []
            
            for path in batch_paths:
                try:
                    if saliency_input:
                        # Load original image, compute saliency, convert to RGB
                        with Image.open(path) as img:
                            img_rgb = np.asarray(img.convert('RGB'), dtype=np.uint8)
                        sal = simple_saliency_map(img_rgb)
                        sal_rgb = saliency_to_rgb(sal)
                        img_sal = Image.fromarray(sal_rgb)
                        img_tensor = transform(img_sal)
                    else:
                        # Normal RGB image
                        with Image.open(path) as img:
                            img = img.convert("RGB")
                        img_tensor = transform(img)
                    
                    imgs_batch.append(img_tensor)
                except Exception as exc:
                    print(f"[WARN] Failed to load '{path}': {exc}. Using zero tensor.")
                    imgs_batch.append(torch.zeros((3, 299, 299), dtype=torch.float32))
            
            if imgs_batch:
                imgs_tensor = torch.stack(imgs_batch).to(device)
                predictions_features.clear()  # Clear previous batch
                with torch.no_grad():
                    _ = model(imgs_tensor)
                
                if predictions_features:
                    batch_features = predictions_features[0]
                    if batch_features.shape[1] != 1000:
                        print(f"[WARN] Predictions layer output has {batch_features.shape[1]} dims, expected 1000")
                    features.append(batch_features)
    
    finally:
        handle.remove()
    
    if features:
        return np.vstack(features)
    return np.zeros((len(image_paths), 1000), dtype=np.float32)


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

    # Load frozen IRv2 models (L1 for original, L2 for saliency)
    print("\n[STEP] Loading frozen pretrained Inception-ResNet-V2 for L1 (original image)")
    model_l1 = load_frozen_model(num_classes=1000)
    if model_l1 is None:
        print("[ERROR] Could not load L1 feature extractor")
        return None
    
    print("[STEP] Loading second frozen pretrained Inception-ResNet-V2 for L2 (saliency)")
    model_l2 = load_frozen_model(num_classes=1000)
    if model_l2 is None:
        print("[ERROR] Could not load L2 feature extractor")
        return None

    # Extract L1 features (original images)
    print(f"\n[STEP] Extracting L1 features (original images) - DF1 (1000-dim)")
    df1_train = extract_features_predictions_layer(model_l1, train_paths, saliency_input=False)
    df1_test = extract_features_predictions_layer(model_l1, test_paths, saliency_input=False)
    print(f"  DF1 train shape: {df1_train.shape}, DF1 test shape: {df1_test.shape}")
    
    # Extract L2 features (saliency maps)
    print(f"[STEP] Extracting L2 features (saliency maps) - DF2 (1000-dim)")
    df2_train = extract_features_predictions_layer(model_l2, train_paths, saliency_input=True)
    df2_test = extract_features_predictions_layer(model_l2, test_paths, saliency_input=True)
    print(f"  DF2 train shape: {df2_train.shape}, DF2 test shape: {df2_test.shape}")
    
    # Free GPU memory
    del model_l1, model_l2
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract L3 handcrafted features (GLCM)
    print(f"[STEP] Extracting L3 handcrafted features (GLCM) - HF (20-dim)")
    glcm_train = extract_all_glcm(train_paths)
    glcm_test = extract_all_glcm(test_paths)
    print(f"  GLCM train shape: {glcm_train.shape}, GLCM test shape: {glcm_test.shape}")

    # Fuse features: [DF1 | DF2 | HF] = 2020-dim (no PCA per paper)
    print(f"\n[STEP] Fusing features: [DF1(1000) | DF2(1000) | HF(20)] = 2020-dim")
    x_train = np.hstack([df1_train, df2_train, glcm_train])
    x_test = np.hstack([df1_test, df2_test, glcm_test])
    print(f"  Fused train shape: {x_train.shape}")
    print(f"  Fused test shape:  {x_test.shape}")
    
    # Normalize (StandardScaler on fused features)
    print(f"[STEP] Normalizing fused features")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print(f"  Normalized train shape: {x_train.shape}")
    print(f"  Normalized test shape:  {x_test.shape}")

    # Train Random Forest (per paper, best classifier)
    print(f"\n[STEP] Training Random Forest classifier (per paper, best performer)")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("DisasterRes-Net-M1-Paper")
    
    with mlflow.start_run(run_name=f"rf_{objective}"):
        clf = RandomForestClassifier(
            n_estimators=100,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
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
    
        model_path = npath(MODEL_DIR, f"rf_{objective}.joblib")
        encoder_path = npath(MODEL_DIR, f"le_{objective}.joblib")
        scaler_path = npath(MODEL_DIR, f"scaler_{objective}.joblib")
        try:
            joblib.dump(clf, model_path)
            joblib.dump(le, encoder_path)
            joblib.dump(scaler, scaler_path)
            
            if objective == "informativeness":
                compat_encoder_path = npath(MODEL_DIR, "le_disaster_type.joblib")
                joblib.dump(le, compat_encoder_path)
                print(f"[INFO] Saved compatibility duplicate label encoder to {compat_encoder_path}")
        except OSError as exc:
            print(f"[ERROR] Failed saving model artifacts: {exc}")
            return None

        # Feature importance (only meaningful for tree-based models)
        df1_names = [f"df1_{i+1}" for i in range(1000)]
        df2_names = [f"df2_{i+1}" for i in range(1000)]
        glcm_names = []
        for ang in ["0", "45", "90", "135"]:
            for prop in ["contrast", "correlation", "energy", "homogeneity", "entropy"]:
                glcm_names.append(f"glcm_{prop}_{ang}")
        feature_names = df1_names + df2_names + glcm_names
        
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
            "classifier": "RandomForest",
            "architecture": "M1_Paper",
            "l1_dim": 1000,
            "l2_dim": 1000,
            "l3_dim": 20,
        }
        metrics_path = _save_metrics_json(objective, metrics)
        
        mlflow.log_param("objective", objective)
        mlflow.log_params({
            "n_estimators": 100,
            "max_features": "sqrt",
            "random_state": 42,
            "classifier": "RandomForestClassifier",
            "architecture": "M1-Paper",
            "fusion_dim": 2020,
            "pca_applied": False,
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
        mlflow.sklearn.log_model(clf, f"rf_model_{objective}")
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(scaler_path)
        if cm_path:
            mlflow.log_artifact(cm_path)
        mlflow.log_artifact(importances_path)
        mlflow.log_artifact(metrics_path)
    
        print(f"\n[INFO] Model saved: {model_path}")
        print(f"[INFO] Label encoder saved: {encoder_path}")
        print(f"[INFO] Scaler saved: {scaler_path}")
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
