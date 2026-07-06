"""Step 5: Baseline comparisons - 6 classifiers × feature sets.

Paper: Gupta & Roy (2025) - Table 2 & 3
Evaluates: KNN, SVM, TB (TreeBagger/ExtraTrees), DT, NB, RF on:
  - Handcrafted (GLCM only)
  - Learned (Deep features only)
  - Fused (Deep + GLCM)

This provides a comprehensive comparison matching the paper's methodology.
"""

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib
import concurrent.futures

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
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
    import torch
    import timm
except Exception as exc:
    torch = None
    timm = None
    print(f"[WARN] PyTorch / timm not available: {exc}")

_SPLIT_DIR = os.path.normpath(os.path.join("split_dataset"))
_MODEL_DIR = os.path.normpath(os.path.join("saved_models"))
_RESULT_DIR = os.path.normpath(os.path.join("results"))
_GLCM_CACHE_DIR = os.path.normpath(os.path.join(".cache"))
_IMG_SIZE = (299, 299)
_BATCH_SIZE = 64
_IMG_EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(_MODEL_DIR, exist_ok=True)
os.makedirs(_RESULT_DIR, exist_ok=True)
os.makedirs(_GLCM_CACHE_DIR, exist_ok=True)

OBJECTIVES = ("informativeness", "damage")

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


def npath(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError as exc:
        print(f"[ERROR] Could not list '{path}': {exc}")
        return []


def _is_image(name: str) -> bool:
    return name.lower().endswith(_IMG_EXTS)


def extract_glcm_features(image_path: str) -> np.ndarray:
    """Extract GLCM features (20-dim) from image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")
            img = img.resize(_IMG_SIZE, Image.Resampling.LANCZOS)
            arr = np.asarray(img)
    except UnidentifiedImageError as exc:
        print(f"[WARN] Corrupt image: '{image_path}': {exc}")
        return np.zeros(20, dtype=np.float32)
    except OSError as exc:
        print(f"[WARN] Could not read '{image_path}': {exc}")
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
    """Extract GLCM for all images with caching."""
    hasher = hashlib.md5()
    for pth in image_paths:
        try:
            mtime = os.path.getmtime(pth)
            hasher.update(f"{pth}_{mtime}".encode('utf-8'))
        except OSError:
            hasher.update(pth.encode('utf-8'))
    
    cache_key = hasher.hexdigest()
    cache_file = npath(_GLCM_CACHE_DIR, f"{cache_key}.npz")
    
    if os.path.exists(cache_file):
        print(f"  [INFO] Loaded GLCM from cache: {cache_file}")
        data = np.load(cache_file)
        return data['arr_0']
        
    print("  [INFO] Extracting GLCM features in parallel...")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    workers = min(8, os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(extract_glcm_features, image_paths),
            total=len(image_paths),
            desc="    GLCM features"
        ))
    feats = np.asarray(results, dtype=np.float32)
    np.savez_compressed(cache_file, feats)
    return feats


def load_deep_features(objective: str, image_paths: Sequence[str], split: str = "test") -> Optional[np.ndarray]:
    """Load cached deep features from step3's PyTorch extraction pipeline."""
    cache_path = npath(_GLCM_CACHE_DIR, "feature_cache", f"{objective}_{split}_features.npz")
    if not os.path.exists(cache_path):
        print(f"  [INFO] Deep feature cache not found: {cache_path}")
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = data["image_paths"].tolist()
        cached_path_to_idx = {path: idx for idx, path in enumerate(cached_paths)}
        try:
            reorder = [cached_path_to_idx[path] for path in image_paths]
        except KeyError:
            print(f"  [WARN] Some paths not found in deep feature cache for {objective}/{split}; skipping.")
            return None

        df1 = data["df1"][reorder]
        df2 = data["df2"][reorder]
        deep = np.hstack([df1, df2])
        print(f"  [INFO] Loaded cached deep features from {cache_path} with shape {deep.shape}")
        return deep
    except Exception as exc:
        print(f"  [WARN] Failed to load deep feature cache '{cache_path}': {exc}")
        return None


def _infer_label_map(objective: str, split_dir: str) -> Dict[str, str]:
    folder_names = [d for d in _safe_listdir(split_dir) if os.path.isdir(npath(split_dir, d))]
    lower = {d.lower() for d in folder_names}

    if objective == "informativeness" and {"informative", "not_informative"}.issubset(lower):
        return {name: name.lower() for name in folder_names}
    if objective == "damage" and {"severe", "mild", "little_or_none"}.issubset(lower):
        return {name: name.lower() for name in folder_names}
    return DEFAULT_LABEL_MAPS[objective]


def load_split(split: str, objective: str) -> Tuple[List[str], List[str]]:
    split_dir = npath(_SPLIT_DIR, split)
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
            print(f"[WARN] Folder '{folder}' not mapped for '{objective}', skipping.")
            continue

        for fname in _safe_listdir(folder_dir):
            if _is_image(fname):
                paths.append(npath(folder_dir, fname))
                labels.append(mapped)

    return paths, labels


def evaluate_classifier(clf, clf_name: str, x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, y_test: np.ndarray, 
                       le: LabelEncoder) -> Dict[str, float]:
    """Train and evaluate a single classifier."""
    print(f"    [{clf_name}]", end=" ", flush=True)
    
    try:
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=skf, scoring='accuracy')
        cv_mean = float(cv_scores.mean())
        
        # Train
        clf.fit(x_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(x_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        
        print(f"Acc={acc:.4f} F1={f1:.4f}")
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_accuracy': cv_mean,
        }
    except Exception as exc:
        print(f"ERROR: {exc}")
        return None


def evaluate_feature_set(objective: str, feature_set: str, 
                        x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, y_test: np.ndarray,
                        le: LabelEncoder) -> Dict[str, Dict]:
    """Evaluate all 6 classifiers on a given feature set."""
    print(f"\n  Feature set: {feature_set}")
    print(f"    Input shape: train={x_train.shape}, test={x_test.shape}")
    
    # Normalize
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Define classifiers
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'SVM': SVC(kernel='rbf', gamma='scale', random_state=42),
        'DT': DecisionTreeClassifier(random_state=42),
        'TB': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'NB': GaussianNB(),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    results = {}
    for clf_name, clf in classifiers.items():
        result = evaluate_classifier(clf, clf_name, x_train_scaled, y_train,
                                     x_test_scaled, y_test, le)
        if result:
            results[clf_name] = result
    
    return results


def run_baseline_comparison(objective: str) -> Optional[Dict[str, Dict]]:
    """Run full baseline comparison for an objective."""
    print("\n" + "=" * 70)
    print(f"STEP 5 - Baseline Comparison - Objective: {objective}")
    print("=" * 70)
    
    # Load data
    train_paths, train_labels = load_split("train", objective)
    test_paths, test_labels = load_split("test", objective)
    
    if not train_paths or not test_paths:
        print("[ERROR] Missing train/test images")
        return None
    
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)
    
    print(f"\n[STEP] Extracting features")
    
    # Extract GLCM (handcrafted)
    print(f"  Extracting GLCM (20-dim)...")
    glcm_train = extract_all_glcm(train_paths)
    glcm_test = extract_all_glcm(test_paths)
    
    # Try to load deep features (from step3)
    print(f"  Loading deep features (1000-dim)...")
    deep_train = load_deep_features(objective, train_paths, split="train")
    deep_test = load_deep_features(objective, test_paths, split="test")
    
    # Evaluate feature sets
    all_results = {}
    
    # 1. Handcrafted only (GLCM)
    print(f"\n[EVAL] Handcrafted features (GLCM only)")
    results_handcrafted = evaluate_feature_set(
        objective, "GLCM-only",
        glcm_train, y_train, glcm_test, y_test, le
    )
    all_results['handcrafted'] = results_handcrafted
    
    # 2. Fused (Deep + GLCM) - if deep features available
    if deep_train is not None and deep_test is not None:
        print(f"\n[EVAL] Fused features (Deep + GLCM)")
        x_train_fused = np.hstack([deep_train, glcm_train])
        x_test_fused = np.hstack([deep_test, glcm_test])
        results_fused = evaluate_feature_set(
            objective, "Deep+GLCM",
            x_train_fused, y_train, x_test_fused, y_test, le
        )
        all_results['fused'] = results_fused
    else:
        print(f"\n[SKIP] Fused evaluation (deep features unavailable)")
    
    return all_results


def save_baseline_report(all_results: Dict[str, Dict], objective: str) -> str:
    """Save comprehensive baseline comparison report."""
    report = {
        'objective': objective,
        'classifiers': list(all_results.get('handcrafted', {}).keys()),
        'feature_sets': list(all_results.keys()),
        'results': all_results,
    }
    
    # Create comparison table
    comparison_rows = []
    for feature_set, clf_results in all_results.items():
        for clf_name, metrics in clf_results.items():
            row = {
                'feature_set': feature_set,
                'classifier': clf_name,
                **metrics
            }
            comparison_rows.append(row)
    
    df = pd.DataFrame(comparison_rows)
    table_path = npath(_RESULT_DIR, f"baseline_comparison_{objective}.csv")
    df.to_csv(table_path, index=False)
    print(f"\n[INFO] Comparison table saved: {table_path}")
    
    # Save JSON report
    report_path = npath(_RESULT_DIR, f"baseline_report_{objective}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Report saved: {report_path}")
    
    return report_path


def main():
    """Main entry point: run baseline comparisons for all objectives."""
    print("\n" + "=" * 70)
    print("STEP 5 - Baseline Comparisons: 6 Classifiers × Feature Sets")
    print("Paper: Gupta & Roy (2025) - Tables 2 & 3")
    print("=" * 70)
    
    all_objectives_results = {}
    for objective in OBJECTIVES:
        results = run_baseline_comparison(objective)
        if results:
            all_objectives_results[objective] = results
            save_baseline_report(results, objective)
    
    print("\n" + "=" * 70)
    print("STEP 5 Complete - Baseline comparisons finished")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
