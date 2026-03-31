"""
Step 3 — feature extraction, compression, ensemble training, dual-head evaluation.
Backend: PyTorch + timm (InceptionResNetV2) — no TensorFlow required.

Fix: class imbalance handled via class_weight dict (RF/ET) and
     scale_pos_weight equivalent (XGB) — no sample_weight fit params needed.
     This avoids sklearn Pipeline nested param forwarding issues.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── matplotlib ───────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except Exception as _e:
    plt = None
    _PLT = False
    print(f"[WARN] matplotlib unavailable: {_e}")

# ── PyTorch + timm ───────────────────────────────────────────────────────────
try:
    import torch
    import timm
    from torchvision import transforms as T

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        _gpu_name = torch.cuda.get_device_name(0)
        _vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[INFO] CUDA GPU : {_gpu_name}  ({_vram_gb:.1f} GB VRAM)")
    else:
        print("[WARN] No CUDA GPU — running on CPU.")
    _TORCH_OK = True
except ImportError as _e:
    _TORCH_OK = False
    print(f"[ERROR] PyTorch/timm unavailable: {_e}")

# ── XGBoost ──────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except ImportError:
    XGBClassifier = None
    _XGB_OK = False
    print("[WARN] XGBoost not installed — ensemble will use RF + ET only.")

# ── Paths & constants ────────────────────────────────────────────────────────

def npath(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))

SPLIT_DIR  = npath("split_dataset")
MODEL_DIR  = npath("saved_models")
RESULT_DIR = npath("results")

IMG_SIZE   = (299, 299)
BATCH_SIZE = 128
CV_FOLDS   = 5
DEEP_DIM   = 1536

OBJECTIVES = ("disaster_type", "damage")
IMG_EXTS   = (".jpg", ".jpeg", ".png")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Label maps ───────────────────────────────────────────────────────────────

DISASTER_TYPE_FOLDERS = {
    "earthquake", "flood", "hurricane",
    "wildfire", "landslide", "not_disaster",
}

DAMAGE_LABEL_MAP: Dict[str, str] = {
    "earthquake":   "severe",
    "hurricane":    "severe",
    "landslide":    "severe",       # fixed: was little_or_none
    "flood":        "mild",
    "wildfire":     "mild",
    "not_disaster": "little_or_none",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError as exc:
        print(f"[ERROR] Cannot list '{path}': {exc}")
        return []

def _is_image(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)

def _load_pil(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError) as exc:
        print(f"[WARN] Skipping '{path}': {exc}")
        return None

# ── Deep feature extractor ───────────────────────────────────────────────────

def build_feature_extractor() -> Optional[object]:
    global DEEP_DIM
    if not _TORCH_OK:
        return None
    try:
        model = timm.create_model(
            "inception_resnet_v2", pretrained=True, num_classes=0)
        model.eval()
        model.to(_DEVICE)

        with torch.no_grad():
            out = model(torch.zeros(1, 3, 299, 299).to(_DEVICE))
        actual_dim = out.shape[-1]
        if actual_dim != DEEP_DIM:
            print(f"[WARN] DEEP_DIM updated {DEEP_DIM} -> {actual_dim}")
            DEEP_DIM = actual_dim
        print(f"[INFO] InceptionResNetV2 ready  dim={actual_dim}  device={_DEVICE}")
        return model
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return None

_NORMALIZE = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def _pil_to_tensor(img: Image.Image) -> "torch.Tensor":
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return _NORMALIZE(torch.from_numpy(arr).permute(2, 0, 1))

def extract_deep_features(
    model: object,
    image_paths: Sequence[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    features: List[np.ndarray] = []
    fallback_count = 0
    model.eval()

    for start in tqdm(range(0, len(image_paths), batch_size),
                      desc="  Deep (InceptionResNetV2)"):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        for pth in batch_paths:
            img = _load_pil(pth)
            if img is None:
                tensors.append(torch.zeros(3, *IMG_SIZE))
                fallback_count += 1
            else:
                tensors.append(_pil_to_tensor(img))

        batch = torch.stack(tensors).to(_DEVICE)
        with torch.no_grad():
            try:
                out = model(batch)
            except RuntimeError as exc:
                print(f"[ERROR] Forward pass failed batch {start}: {exc}")
                print("[HINT]  Reduce BATCH_SIZE if OOM.")
                out = torch.zeros(len(batch_paths), DEEP_DIM)
        features.append(out.cpu().numpy().astype(np.float32))

    if fallback_count:
        print(f"[WARN] {fallback_count} images used zero fallback.")

    result = np.vstack(features)
    if result.shape[1] != DEEP_DIM:
        raise ValueError(f"Dim mismatch: got {result.shape[1]}, expected {DEEP_DIM}")
    return result

# ── GLCM (20-dim) ────────────────────────────────────────────────────────────

_GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_DIM     = len(_GLCM_ANGLES) * 5

def extract_glcm_features(image_path: str) -> np.ndarray:
    try:
        with Image.open(image_path) as img:
            arr = np.clip(
                (np.asarray(img.convert("L").resize(
                    IMG_SIZE, Image.Resampling.LANCZOS)) / 32
                ).astype(np.uint8), 0, 7)
    except (UnidentifiedImageError, OSError):
        return np.zeros(GLCM_DIM, dtype=np.float32)

    out: List[float] = []
    for angle in _GLCM_ANGLES:
        glcm = graycomatrix(arr, distances=[1], angles=[angle],
                            levels=8, symmetric=True, normed=True)
        out.append(float(graycoprops(glcm, "contrast")[0, 0]))
        out.append(float(graycoprops(glcm, "correlation")[0, 0]))
        out.append(float(graycoprops(glcm, "energy")[0, 0]))
        out.append(float(graycoprops(glcm, "homogeneity")[0, 0]))
        pmat = glcm[:, :, 0, 0]
        out.append(float(-np.sum(pmat * np.log2(np.where(pmat > 0, pmat, 1e-10)))))
    return np.asarray(out, dtype=np.float32)

# ── LBP (59-dim) ─────────────────────────────────────────────────────────────

LBP_DIM = 59

def extract_lbp_features(image_path: str) -> np.ndarray:
    try:
        with Image.open(image_path) as img:
            arr = np.asarray(
                img.convert("L").resize(IMG_SIZE, Image.Resampling.LANCZOS),
                dtype=np.uint8)
    except (UnidentifiedImageError, OSError):
        return np.zeros(LBP_DIM, dtype=np.float32)

    lbp = local_binary_pattern(arr, P=8, R=1, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    if len(hist) < LBP_DIM:
        hist = np.concatenate([hist, np.zeros(LBP_DIM - len(hist))])
    return hist[:LBP_DIM].astype(np.float32)

# ── HSV (96-dim) ─────────────────────────────────────────────────────────────

HSV_DIM = 32 * 3

def extract_hsv_features(image_path: str) -> np.ndarray:
    try:
        with Image.open(image_path) as img:
            arr = np.asarray(
                img.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS),
                dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError):
        return np.zeros(HSV_DIM, dtype=np.float32)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    diff = cmax - np.minimum(np.minimum(r, g), b) + 1e-8
    h = np.where(cmax == r, (60*((g-b)/diff) % 360),
          np.where(cmax == g, 60*((b-r)/diff)+120,
                               60*((r-g)/diff)+240)) / 360.0
    s = np.where(cmax == 0, 0.0, diff / (cmax + 1e-8))
    out = []
    for ch in (h, s, cmax):
        hist, _ = np.histogram(ch.ravel(), bins=32, range=(0,1), density=True)
        out.append(hist.astype(np.float32))
    return np.concatenate(out)

# ── Batch wrappers ───────────────────────────────────────────────────────────

def extract_all_glcm(paths: Sequence[str]) -> np.ndarray:
    return np.asarray([extract_glcm_features(p)
                       for p in tqdm(paths, desc="  GLCM")], dtype=np.float32)

def extract_all_lbp(paths: Sequence[str]) -> np.ndarray:
    return np.asarray([extract_lbp_features(p)
                       for p in tqdm(paths, desc="  LBP")], dtype=np.float32)

def extract_all_hsv(paths: Sequence[str]) -> np.ndarray:
    return np.asarray([extract_hsv_features(p)
                       for p in tqdm(paths, desc="  HSV")], dtype=np.float32)

# ── Dataset loading ──────────────────────────────────────────────────────────

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
                print(f"[WARN] Unknown folder '{folder}' — skipping.")
                continue
            label = folder_lower
        elif objective == "damage":
            label = DAMAGE_LABEL_MAP.get(folder_lower)
            if label is None:
                print(f"[WARN] No damage mapping for '{folder}' — skipping.")
                continue
        else:
            print(f"[ERROR] Unknown objective '{objective}'.")
            return [], []

        for fname in _safe_listdir(folder_dir):
            if _is_image(fname):
                paths.append(npath(folder_dir, fname))
                labels.append(label)

    print(f"[INFO] {split:5s} | {objective:15s} | "
          f"{len(paths)} images, {len(set(labels))} classes")
    return paths, labels

# ── Class weight helpers ─────────────────────────────────────────────────────

def get_class_weight_dict(labels: List[str], le: LabelEncoder) -> Dict[int, float]:
    """
    Returns {encoded_class_int: weight} for use in RF/ET class_weight param.
    Weights are computed from actual label distribution — balanced formula.
    """
    w = compute_class_weight("balanced", classes=le.classes_, y=np.array(labels))
    weight_dict = {i: float(w[i]) for i in range(len(le.classes_))}

    print("[INFO] Class weights:")
    for i, cls in enumerate(le.classes_):
        print(f"         {cls:<20}  {weight_dict[i]:.4f}x")
    return weight_dict


def get_xgb_sample_weights(
    labels: List[str],
    le: LabelEncoder,
    class_weight_dict: Dict[int, float],
) -> np.ndarray:
    """
    XGBoost multiclass does not support class_weight= directly.
    We convert it to per-sample weights instead, used only in XGB.fit().
    XGB is trained separately (outside the Pipeline) then added to VotingClassifier.
    """
    y_enc = le.transform(labels)
    return np.array([class_weight_dict[y] for y in y_enc], dtype=np.float32)

# ── Pipeline builder ─────────────────────────────────────────────────────────

def build_pipeline(class_weight_dict: Dict[int, float]) -> Pipeline:
    """
    RF and ET accept class_weight= natively — no fit params needed.
    XGB is trained separately; see train_and_evaluate() for how it is added
    to the VotingClassifier after fitting.

    This is the cleanest approach compatible with all recent sklearn versions:
    - No sample_weight threading through Pipeline
    - No nested clf__estimator__param syntax
    - Weights baked into the estimator constructors directly
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight_dict,   # weights baked in — no fit param needed
        random_state=42,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight_dict,   # weights baked in — no fit param needed
        random_state=43,
        n_jobs=-1,
    )
    estimators = [("rf", rf), ("et", et)]

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=0.95, svd_solver="full", random_state=42)),
        ("clf",    VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)),
    ])


def _fit_xgb_on_compressed(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    xgb_sample_weights: np.ndarray,
    fitted_pipeline: Pipeline,
) -> "XGBClassifier":
    """
    Transforms train features through the already-fitted scaler+PCA,
    then fits XGBoost with per-sample weights on the compressed features.
    Returns the fitted XGBClassifier.
    """
    # Use only scaler+PCA steps to transform — do not run through clf
    scaler = fitted_pipeline.named_steps["scaler"]
    pca    = fitted_pipeline.named_steps["pca"]
    x_compressed = pca.transform(scaler.transform(x_train_raw))

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=44,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(x_compressed, y_train, sample_weight=xgb_sample_weights)
    return xgb


def _add_xgb_to_voting(
    fitted_pipeline: Pipeline,
    xgb: "XGBClassifier",
) -> Pipeline:
    """
    Replaces the VotingClassifier inside the pipeline with a new one that
    includes the pre-fitted XGBClassifier alongside the already-fitted RF/ET.

    We set estimators to None so VotingClassifier skips re-fitting them,
    then manually inject the pre-fitted estimators.
    """
    old_voting: VotingClassifier = fitted_pipeline.named_steps["clf"]

    # In newer sklearn, estimators_ is a plain list of fitted objects,
    # NOT (name, estimator) tuples. Use named_estimators_ for name-based lookup.
    fitted_rf = old_voting.named_estimators_["rf"]
    fitted_et = old_voting.named_estimators_["et"]

    new_voting = VotingClassifier(
        estimators=[
            ("rf",  fitted_rf),
            ("et",  fitted_et),
            ("xgb", xgb),
        ],
        voting="soft",
        n_jobs=1,
    )
    # Mark as fitted so sklearn does not try to re-fit it on predict
    new_voting.estimators_       = [fitted_rf, fitted_et, xgb]
    new_voting.le_               = old_voting.le_
    new_voting.classes_          = old_voting.classes_
    new_voting.named_estimators_ = {"rf": fitted_rf, "et": fitted_et, "xgb": xgb}

    fitted_pipeline.steps[-1] = ("clf", new_voting)
    return fitted_pipeline

# ── Cross-validation ─────────────────────────────────────────────────────────

def run_cv(
    x: np.ndarray,
    y: np.ndarray,
    class_weight_dict: Dict[int, float],
    xgb_sample_weights: np.ndarray,
    objective: str,
) -> Dict[str, float]:
    print(f"\n[CV] {CV_FOLDS}-fold stratified CV — {objective}")
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores: List[float] = []

    for fold, (tr, val) in enumerate(skf.split(x, y), 1):
        # Build and fit RF+ET pipeline (weights baked in via class_weight_dict)
        pipe = build_pipeline(class_weight_dict)
        pipe.fit(x[tr], y[tr])

        # Fit XGBoost separately with sample weights on compressed features
        if _XGB_OK:
            xgb = _fit_xgb_on_compressed(x[tr], y[tr], xgb_sample_weights[tr], pipe)
            pipe = _add_xgb_to_voting(pipe, xgb)

        f1 = f1_score(y[val], pipe.predict(x[val]),
                      average="weighted", zero_division=0)
        scores.append(f1)
        print(f"  Fold {fold}/{CV_FOLDS}  weighted-F1 = {f1:.4f}")

    mean, std = float(np.mean(scores)), float(np.std(scores))
    print(f"[CV] Result: {mean:.4f} +/- {std:.4f}")
    return {"cv_mean_f1": mean, "cv_std_f1": std}

# ── Persistence helpers ──────────────────────────────────────────────────────

def _save_metrics(objective: str, metrics: Dict) -> None:
    path = npath(RESULT_DIR, f"metrics_{objective}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAVED] {path}")
    except OSError as exc:
        print(f"[WARN] Could not save metrics: {exc}")

def _save_confusion_plot(
    cm: np.ndarray, labels: Sequence[str], objective: str
) -> None:
    if not _PLT:
        return
    path = npath(RESULT_DIR, f"confusion_matrix_{objective}.png")
    try:
        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion matrix — {objective}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {path}")
    except OSError as exc:
        print(f"[WARN] Could not save confusion plot: {exc}")

# ── Main training function ───────────────────────────────────────────────────

def train_and_evaluate(objective: str, extractor: object) -> Optional[Dict]:
    print("\n" + "=" * 72)
    print(f"  OBJECTIVE : {objective}")
    print("=" * 72)

    train_paths, train_labels = load_split("train", objective)
    test_paths,  test_labels  = load_split("test",  objective)
    if not train_paths or not test_paths:
        print("[ERROR] Empty split — run step2_preprocess.py first.")
        return None

    le = LabelEncoder()
    le.fit(train_labels + test_labels)
    y_train = le.transform(train_labels)
    y_test  = le.transform(test_labels)
    print(f"[INFO] Classes : {list(le.classes_)}")

    # Compute weights — used as class_weight dict (RF/ET) and sample weights (XGB)
    class_weight_dict   = get_class_weight_dict(train_labels, le)
    xgb_sample_weights  = get_xgb_sample_weights(train_labels, le, class_weight_dict)

    # Feature extraction
    print("\n[FEAT] Train features...")
    x_train = np.hstack([
        extract_deep_features(extractor, train_paths),
        extract_all_glcm(train_paths),
        extract_all_lbp(train_paths),
        extract_all_hsv(train_paths),
    ])

    print("\n[FEAT] Test features...")
    x_test = np.hstack([
        extract_deep_features(extractor, test_paths),
        extract_all_glcm(test_paths),
        extract_all_lbp(test_paths),
        extract_all_hsv(test_paths),
    ])
    print(f"[INFO] Fused feature dim : {x_train.shape[1]}")

    # Cross-validation
    cv_metrics = run_cv(
        x_train, y_train, class_weight_dict, xgb_sample_weights, objective
    )

    # Final fit — RF+ET via class_weight in constructor, XGB via sample_weight
    print("\n[TRAIN] Fitting final pipeline (RF + ET)...")
    pipe = build_pipeline(class_weight_dict)
    pipe.fit(x_train, y_train)  # clean fit — no sample_weight params needed

    pca: PCA = pipe.named_steps["pca"]
    print(f"[INFO] PCA : {x_train.shape[1]}-dim -> {pca.n_components_}-dim  "
          f"({pca.explained_variance_ratio_.sum() * 100:.1f}% variance retained)")

    if _XGB_OK:
        print("[TRAIN] Fitting XGBoost on compressed features...")
        xgb  = _fit_xgb_on_compressed(x_train, y_train, xgb_sample_weights, pipe)
        pipe = _add_xgb_to_voting(pipe, xgb)
        print("[INFO] XGBoost added to ensemble.")

    # Evaluate
    y_pred    = pipe.predict(x_test)
    acc       = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred,
                                       average="weighted", zero_division=0))
    recall    = float(recall_score(y_test, y_pred,
                                    average="weighted", zero_division=0))
    f1        = float(f1_score(y_test, y_pred,
                                average="weighted", zero_division=0))

    print(f"\n[RESULT] acc={acc:.4f}  prec={precision:.4f}  "
          f"rec={recall:.4f}  f1={f1:.4f}")
    print(classification_report(y_test, y_pred,
                                  target_names=le.classes_, zero_division=0))

    # Save
    joblib.dump(pipe, npath(MODEL_DIR, f"pipeline_{objective}.joblib"))
    joblib.dump(le,   npath(MODEL_DIR, f"le_{objective}.joblib"))
    print(f"[SAVED] pipeline_{objective}.joblib")
    print(f"[SAVED] le_{objective}.joblib")

    _save_confusion_plot(confusion_matrix(y_test, y_pred), le.classes_, objective)

    metrics = {
        "objective":       objective,
        "accuracy":        acc,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "pca_components":  int(pca.n_components_),
        "raw_feature_dim": int(x_train.shape[1]),
        "train_images":    len(train_paths),
        "test_images":     len(test_paths),
        **cv_metrics,
    }
    _save_metrics(objective, metrics)
    return metrics

# ── Inference helper ─────────────────────────────────────────────────────────

def predict_single_image(image_path: str, extractor: object) -> Dict[str, str]:
    """
    Run both objectives on one image.
    Returns e.g.:
      {'disaster_type': 'landslide', 'disaster_type_confidence': '0.912',
       'damage': 'severe',           'damage_confidence': '0.887'}
    """
    img = _load_pil(image_path)
    if img is None:
        return {"error": f"Could not load {image_path}"}

    t = _pil_to_tensor(img).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        deep = extractor(t).cpu().numpy().astype(np.float32)

    x = np.hstack([
        deep,
        extract_glcm_features(image_path)[np.newaxis],
        extract_lbp_features(image_path)[np.newaxis],
        extract_hsv_features(image_path)[np.newaxis],
    ])

    results: Dict[str, str] = {}
    for obj in OBJECTIVES:
        pipe_path = npath(MODEL_DIR, f"pipeline_{obj}.joblib")
        le_path   = npath(MODEL_DIR, f"le_{obj}.joblib")
        if not os.path.exists(pipe_path):
            results[obj] = "model_not_trained"
            continue
        pipe  = joblib.load(pipe_path)
        le    = joblib.load(le_path)
        pred  = le.inverse_transform(pipe.predict(x))[0]
        conf  = float(np.max(pipe.predict_proba(x)[0]))
        results[obj]                 = pred
        results[f"{obj}_confidence"] = f"{conf:.3f}"

    return results

# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    extractor = build_feature_extractor()
    if extractor is None:
        print("[ERROR] Feature extractor unavailable.")
        print("[HINT]  pip install torch torchvision timm")
        return 1

    summary: List[Dict] = []
    all_ok = True

    for objective in OBJECTIVES:
        result = train_and_evaluate(objective, extractor)
        if result is None:
            all_ok = False
        else:
            summary.append(result)

    if summary:
        print("\n" + "=" * 72)
        print(f"  {'Objective':<20} {'Acc':>7} {'F1':>7} {'CV F1':>16} {'PCA dim':>8}")
        print("-" * 64)
        for m in summary:
            cv = f"{m['cv_mean_f1']:.4f}+/-{m['cv_std_f1']:.4f}"
            print(f"  {m['objective']:<20} {m['accuracy']:>7.4f} "
                  f"{m['f1']:>7.4f} {cv:>16} {m['pca_components']:>8}")
        print("=" * 72)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())