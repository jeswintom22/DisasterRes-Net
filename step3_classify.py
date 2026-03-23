"""Step 3: extract features, train Random Forest, evaluate, and save models."""

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
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
from sklearn.preprocessing import LabelEncoder
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    print(f"[WARN] matplotlib import failed: {exc}. Confusion image generation will be disabled.")

try:
    from tensorflow.keras.applications import InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    from tensorflow.keras.models import Model
except Exception as exc:
    InceptionResNetV2 = None
    preprocess_input = None
    Model = None
    print(f"[ERROR] TensorFlow import failed: {exc}")
    print("[ERROR] Install TensorFlow for Python 3.10 (Windows), then retry.")


def npath(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


SPLIT_DIR = npath("split_dataset")
MODEL_DIR = npath("saved_models")
RESULT_DIR = npath("results")
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

OBJECTIVES = ("informativeness", "damage")
IMG_EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


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


def build_feature_extractor() -> Optional[object]:
    if InceptionResNetV2 is None or preprocess_input is None or Model is None:
        return None
    try:
        base = InceptionResNetV2(weights="imagenet", include_top=True, input_shape=(299, 299, 3))
        model = Model(inputs=base.input, outputs=base.get_layer("predictions").output)
        model.trainable = False
        return model
    except Exception as exc:
        print(f"[ERROR] Failed to build InceptionResNetV2 extractor: {exc}")
        return None


def _load_rgb_299(path: str) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.float32)
    except UnidentifiedImageError as exc:
        print(f"[WARN] Corrupt image skipped: '{path}'. Reason: {exc}")
        return None
    except OSError as exc:
        print(f"[WARN] Could not read image '{path}': {exc}")
        return None


def extract_deep_features(model: object, image_paths: Sequence[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    features: List[np.ndarray] = []
    fallback_count = 0
    for start in tqdm(range(0, len(image_paths), batch_size), desc="  Deep features"):
        batch_paths = image_paths[start : start + batch_size]
        batch_imgs: List[np.ndarray] = []
        for pth in batch_paths:
            arr = _load_rgb_299(pth)
            if arr is None:
                arr = np.zeros((299, 299, 3), dtype=np.float32)
                fallback_count += 1
            batch_imgs.append(arr)
        batch_array = np.asarray(batch_imgs, dtype=np.float32)
        batch_array = preprocess_input(batch_array)
        try:
            preds = model.predict(batch_array, verbose=0)
        except Exception as exc:
            print(f"[ERROR] Deep feature extraction failed for batch starting at {start}: {exc}")
            preds = np.zeros((len(batch_paths), 1000), dtype=np.float32)
        features.extend(preds)

    if fallback_count:
        print(f"[WARN] Deep features used blank fallback for {fallback_count} unreadable images.")
    return np.asarray(features, dtype=np.float32)


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
    feats: List[np.ndarray] = []
    for pth in tqdm(image_paths, desc="  GLCM features"):
        feats.append(extract_glcm_features(pth))
    return np.asarray(feats, dtype=np.float32)


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

    extractor = build_feature_extractor()
    if extractor is None:
        print("[ERROR] Feature extractor unavailable. Install TensorFlow and retry.")
        return None

    train_paths, train_labels = load_split("train", objective)
    test_paths, test_labels = load_split("test", objective)

    if not train_paths or not test_paths:
        print("[ERROR] Missing train/test image paths. Run step2_preprocess.py first.")
        return None

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)

    deep_train = extract_deep_features(extractor, train_paths)
    deep_test = extract_deep_features(extractor, test_paths)
    glcm_train = extract_all_glcm(train_paths)
    glcm_test = extract_all_glcm(test_paths)

    if deep_train.shape[1] != 1000:
        print(f"[WARN] Deep feature dim is {deep_train.shape[1]}, expected 1000.")
    if glcm_train.shape[1] != 20:
        print(f"[WARN] GLCM feature dim is {glcm_train.shape[1]}, expected 20.")

    x_train = np.hstack([deep_train, glcm_train])
    x_test = np.hstack([deep_test, glcm_test])
    print(f"[INFO] Fused feature shape train: {x_train.shape}")
    print(f"[INFO] Fused feature shape test : {x_test.shape}")

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    try:
        clf.fit(x_train, y_train)
    except Exception as exc:
        print(f"[ERROR] RandomForest training failed: {exc}")
        return None

    y_pred = clf.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred)

    print(f"[RESULT] accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    print("[REPORT]")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    cm_path = _save_confusion_plot(cm, le.classes_, objective)

    model_path = npath(MODEL_DIR, f"rf_{objective}.joblib")
    encoder_path = npath(MODEL_DIR, f"le_{objective}.joblib")
    try:
        joblib.dump(clf, model_path)
        joblib.dump(le, encoder_path)
    except OSError as exc:
        print(f"[ERROR] Failed saving model artifacts: {exc}")
        return None

    metrics = {
        "objective": objective,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_images": float(len(train_paths)),
        "test_images": float(len(test_paths)),
        "feature_dim": float(x_train.shape[1]),
    }
    metrics_path = _save_metrics_json(objective, metrics)

    print(f"[INFO] Model saved: {model_path}")
    print(f"[INFO] Label encoder saved: {encoder_path}")
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
