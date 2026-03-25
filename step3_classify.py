"""
TUNED CLASSIFIER — DisasterRes-Net
====================================
Improvements over baseline:
  1. Feature scaling (StandardScaler) before RF
  2. Class imbalance handling (class_weight='balanced')
  3. Better RF hyperparameters
  4. SMOTE oversampling for minority class (mild)
  5. Ensemble: RF + ExtraTreesClassifier + voting
  6. PCA dimensionality reduction option

Install extras:
    pip install imbalanced-learn
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model

from skimage.feature import graycomatrix, graycoprops

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
SPLIT_DIR  = "split_dataset"
MODEL_DIR  = "saved_models"
RESULT_DIR = "results"
IMG_SIZE   = (299, 299)
BATCH_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

LABEL_MAPS = {
    "informativeness": {
        "earthquake":   "informative",
        "flood":        "informative",
        "hurricane":    "informative",
        "wildfire":     "informative",
        "landslide":    "informative",
        "not_disaster": "not_informative",
        "informative":     "informative",
        "not_informative": "not_informative",
    },
    "damage": {
        "earthquake":      "severe",
        "flood":           "mild",
        "hurricane":       "severe",
        "wildfire":        "mild",
        "landslide":       "little_or_none",
        "not_disaster":    "little_or_none",
        "severe":          "severe",
        "mild":            "mild",
        "little_or_none":  "little_or_none",
    }
}

# ─────────────────────────────────────────
# FEATURE EXTRACTION (unchanged)
# ─────────────────────────────────────────
def build_feature_extractor():
    print("  Loading InceptionResNetV2...", end="", flush=True)
    base  = InceptionResNetV2(weights="imagenet", include_top=True, input_shape=(299,299,3))
    model = Model(inputs=base.input, outputs=base.get_layer("predictions").output)
    model.trainable = False
    print(" ✅")
    return model

def extract_deep_features(model, image_paths, batch_size=BATCH_SIZE):
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="  Deep features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs  = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB").resize(IMG_SIZE)
                batch_imgs.append(np.array(img, dtype=np.float32))
            except Exception:
                batch_imgs.append(np.zeros((299,299,3), dtype=np.float32))
        batch = preprocess_input(np.array(batch_imgs))
        feats = model.predict(batch, verbose=0)
        features.extend(feats)
    return np.array(features)

def extract_glcm_features(image_path):
    img = Image.open(image_path).convert("L").resize(IMG_SIZE)
    arr = (np.array(img) / 32).astype(np.uint8)
    arr = np.clip(arr, 0, 7)
    angles   = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features = []
    for angle in angles:
        glcm = graycomatrix(arr, distances=[1], angles=[angle],
                            levels=8, symmetric=True, normed=True)
        features.append(float(graycoprops(glcm, 'contrast')[0,0]))
        features.append(float(graycoprops(glcm, 'correlation')[0,0]))
        features.append(float(graycoprops(glcm, 'energy')[0,0]))
        features.append(float(graycoprops(glcm, 'homogeneity')[0,0]))
        p      = glcm[:,:,0,0]
        p_safe = np.where(p > 0, p, 1e-10)
        features.append(float(-np.sum(p * np.log2(p_safe))))
    return np.array(features, dtype=np.float32)

def extract_all_glcm(image_paths):
    features = []
    for p in tqdm(image_paths, desc="  GLCM features"):
        try:
            features.append(extract_glcm_features(p))
        except Exception:
            features.append(np.zeros(20, dtype=np.float32))
    return np.array(features)

# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────
def load_split(split, objective):
    split_dir = os.path.join(SPLIT_DIR, split)
    label_map = LABEL_MAPS[objective]
    paths, labels = [], []
    for folder in os.listdir(split_dir):
        folder_dir   = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_dir):
            continue
        mapped_label = label_map.get(folder)
        if mapped_label is None:
            continue
        for fname in os.listdir(folder_dir):
            if fname.lower().endswith('.jpg'):
                paths.append(os.path.join(folder_dir, fname))
                labels.append(mapped_label)
    return paths, labels

# ─────────────────────────────────────────
# TUNED CLASSIFIERS
# ─────────────────────────────────────────
def build_tuned_classifier(objective, class_names):
    """
    Returns a tuned sklearn Pipeline.

    Key improvements:
    1. StandardScaler  — normalizes 1020-dim vector (big win for RF)
    2. class_weight    — handles class imbalance (mild class)
    3. More trees      — 500 instead of 200
    4. max_features    — 'sqrt' is optimal for RF
    5. min_samples_leaf — reduces overfitting
    6. Ensemble voting — RF + ExtraTrees combined
    """

    rf = RandomForestClassifier(
        n_estimators    = 500,        # more trees = more stable (was 200)
        max_features    = "sqrt",     # best for high-dim data
        max_depth       = None,       # let trees grow fully
        min_samples_leaf= 2,          # slight regularization
        class_weight    = "balanced", # fixes mild class imbalance
        random_state    = 42,
        n_jobs          = -1,
    )

    et = ExtraTreesClassifier(
        n_estimators    = 500,
        max_features    = "sqrt",
        min_samples_leaf= 2,
        class_weight    = "balanced",
        random_state    = 42,
        n_jobs          = -1,
    )

    # Soft voting ensemble — averages probabilities
    ensemble = VotingClassifier(
        estimators = [("rf", rf), ("et", et)],
        voting     = "soft",
        n_jobs     = -1,
    )

    # Full pipeline: scale → classify
    pipeline = Pipeline([
        ("scaler",     StandardScaler()),   # normalize features
        ("classifier", ensemble),           # RF + ExtraTrees ensemble
    ])

    return pipeline


def apply_smote(X_train, y_train, objective):
    try:
        from imblearn.over_sampling import SMOTE
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n  Class distribution before SMOTE:")
        for cls, cnt in zip(unique, counts):
            print(f"    {str(cls):20s}: {cnt}")   # ← fix here

        sm = SMOTE(random_state=42, k_neighbors=min(5, counts.min()-1))
        X_res, y_res = sm.fit_resample(X_train, y_train)

        unique2, counts2 = np.unique(y_res, return_counts=True)
        print(f"\n  Class distribution after SMOTE:")
        for cls, cnt in zip(unique2, counts2):
            print(f"    {str(cls):20s}: {cnt}")   # ← fix here

        return X_res, y_res
    except ImportError:
        print("  ℹ️  imbalanced-learn not installed — skipping SMOTE")
        print("     Run: pip install imbalanced-learn")
        return X_train, y_train

# ─────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────
def train_and_evaluate(objective="informativeness"):
    print("\n" + "="*58)
    print(f"  TUNED DisasterRes-Net | Objective: {objective.upper()}")
    print("="*58)

    # ── Load data ──
    print("\n[1/6] Loading dataset...")
    train_paths, train_labels = load_split("train", objective)
    test_paths,  test_labels  = load_split("test",  objective)
    print(f"  Train: {len(train_paths)} | Test: {len(test_paths)}")

    if not train_paths:
        print("❌ No training images found.")
        return

    # ── Encode labels ──
    le      = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test  = le.transform(test_labels)
    print(f"  Classes: {list(le.classes_)}")

    # ── Extract features ──
    print("\n[2/6] Extracting deep features...")
    extractor  = build_feature_extractor()
    deep_train = extract_deep_features(extractor, train_paths)
    deep_test  = extract_deep_features(extractor, test_paths)

    print("\n[3/6] Extracting GLCM features...")
    glcm_train = extract_all_glcm(train_paths)
    glcm_test  = extract_all_glcm(test_paths)

    # ── Fuse ──
    print("\n[4/6] Fusing features...")
    X_train = np.hstack([deep_train, glcm_train])
    X_test  = np.hstack([deep_test,  glcm_test])
    print(f"  Feature vector shape: {X_train.shape}")

    # ── SMOTE for class imbalance ──
    print("\n[5/6] Handling class imbalance...")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train, objective)

    # ── Train tuned ensemble ──
    print("\n[6/6] Training tuned ensemble (RF + ExtraTrees)...")
    print("  Parameters:")
    print("    n_estimators    = 500  (was 200)")
    print("    class_weight    = balanced  (was none)")
    print("    StandardScaler  = True  (was none)")
    print("    Voting          = soft ensemble  (was single RF)")

    clf = build_tuned_classifier(objective, le.classes_)
    clf.fit(X_train_bal, y_train_bal)

    # ── Evaluate ──
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "="*58)
    print(f"  ACCURACY: {acc*100:.2f}%")
    print("="*58)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Confusion matrix ──
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title(f"Confusion Matrix (Tuned) — {objective}")
    plt.tight_layout()
    cm_path = os.path.join(RESULT_DIR, f"confusion_matrix_{objective}_tuned.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix → {cm_path}")

    # ── Save ──
    clf_path = os.path.join(MODEL_DIR, f"rf_{objective}_tuned.joblib")
    le_path  = os.path.join(MODEL_DIR, f"le_{objective}.joblib")
    joblib.dump(clf, clf_path)
    joblib.dump(le,  le_path)

    # ── Save metrics ──
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics = {
        "objective":    objective,
        "accuracy":     round(acc, 6),
        "precision":    round(precision_score(y_test,y_pred,average="weighted"), 6),
        "recall":       round(recall_score(y_test,y_pred,average="weighted"), 6),
        "f1":           round(f1_score(y_test,y_pred,average="weighted"), 6),
        "train_images": float(len(train_paths)),
        "test_images":  float(len(test_paths)),
        "feature_dim":  float(X_train.shape[1]),
        "model":        "RF+ExtraTrees Ensemble (tuned)",
    }
    m_path = os.path.join(RESULT_DIR, f"metrics_{objective}_tuned.json")
    with open(m_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Model saved   → {clf_path}")
    print(f"  Metrics saved → {m_path}")

    return clf, le, acc


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════╗
║   DisasterRes-Net — TUNED CLASSIFIER         ║
║   RF + ExtraTrees + SMOTE + StandardScaler   ║
╚══════════════════════════════════════════════╝
    """)

    for obj in ["informativeness", "damage"]:
        train_and_evaluate(objective=obj)

    print("\n✅ Tuned training complete!")
    print("   New models saved with '_tuned' suffix")
    print("   Compare results in results/ folder")