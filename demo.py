"""
LIVE DEMO — DisasterRes-Net
============================
Run this in front of your guide to demonstrate
the trained model predicting on a new image.

Usage:
    python demo.py                    ← uses built-in test images
    python demo.py my_image.jpg       ← predict on your own image
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import os
import sys
import joblib
import numpy as np
from PIL import Image

# ── Try importing TensorFlow quietly ──
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.applications import InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    from tensorflow.keras.models import Model
    TF_OK = True
except Exception:
    TF_OK = False

from skimage.feature import graycomatrix, graycoprops

IMG_SIZE    = (299, 299)
MODEL_DIR   = "saved_models"
RESULTS_DIR = "results"

# ── Sample test images from your dataset ──
# Script will auto-find one if no argument given
SAMPLE_DIRS = [
    os.path.join("split_dataset", "test", "severe"),
    os.path.join("split_dataset", "test", "informative"),
    os.path.join("split_dataset", "test", "earthquake"),
    os.path.join("split_dataset", "test", "flood"),
]


def banner(text, char="═"):
    w = 55
    print("\n" + char * w)
    print(f"  {text}")
    print(char * w)


def find_sample_image():
    """Auto-find a test image from the dataset."""
    for d in SAMPLE_DIRS:
        if os.path.exists(d):
            imgs = [f for f in os.listdir(d)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                return os.path.join(d, imgs[0])
    # Last resort — search split_dataset
    for root, dirs, files in os.walk("split_dataset"):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, f)
    return None


def load_and_show_image(image_path):
    """Load image and print basic info."""
    img = Image.open(image_path).convert("RGB")
    print(f"\n  📸 Image     : {os.path.basename(image_path)}")
    print(f"  📁 Path      : {image_path}")
    print(f"  📐 Size      : {img.width} × {img.height} px")
    return img


def build_extractor():
    """Build InceptionResNetV2 feature extractor."""
    print("\n  ⏳ Loading InceptionResNetV2...", end="", flush=True)
    base  = InceptionResNetV2(weights="imagenet", include_top=True,
                               input_shape=(299, 299, 3))
    model = Model(inputs=base.input,
                  outputs=base.get_layer("predictions").output)
    model.trainable = False
    print(" ✅ Ready")
    return model


def extract_deep(model, image_path):
    img  = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr  = np.array(img, dtype=np.float32)
    batch = preprocess_input(np.expand_dims(arr, 0))
    return model.predict(batch, verbose=0)[0]


def extract_glcm(image_path):
    img = Image.open(image_path).convert("L").resize(IMG_SIZE)
    arr = (np.array(img) / 32).astype(np.uint8)
    arr = np.clip(arr, 0, 7)
    angles   = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features = []
    for angle in angles:
        glcm = graycomatrix(arr, distances=[1], angles=[angle],
                            levels=8, symmetric=True, normed=True)
        features.append(float(graycoprops(glcm, 'contrast')[0, 0]))
        features.append(float(graycoprops(glcm, 'correlation')[0, 0]))
        features.append(float(graycoprops(glcm, 'energy')[0, 0]))
        features.append(float(graycoprops(glcm, 'homogeneity')[0, 0]))
        p      = glcm[:, :, 0, 0]
        p_safe = np.where(p > 0, p, 1e-10)
        features.append(float(-np.sum(p * np.log2(p_safe))))
    return np.array(features, dtype=np.float32)


def predict(image_path, objective, extractor):
    """Run prediction for one objective."""
    clf_path = os.path.join(MODEL_DIR, f"rf_{objective}.joblib")
    le_path  = os.path.join(MODEL_DIR, f"le_{objective}.joblib")

    if not os.path.exists(clf_path):
        print(f"  ⚠️  Model not found: {clf_path}")
        return

    clf = joblib.load(clf_path)
    le  = joblib.load(le_path)

    deep = extract_deep(extractor, image_path)
    glcm = extract_glcm(image_path)
    X    = np.hstack([deep, glcm]).reshape(1, -1)

    pred_idx   = clf.predict(X)[0]
    pred_proba = clf.predict_proba(X)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = pred_proba.max() * 100

    # ── Display result ──
    obj_name = "Informativeness" if objective == "informativeness" else "Damage Severity"
    print(f"\n  {'─'*48}")
    print(f"  🎯  {obj_name}")
    print(f"  {'─'*48}")
    print(f"  Prediction : {pred_label.upper().replace('_', ' ')}")
    print(f"  Confidence : {confidence:.1f}%")
    print()

    # Bar chart in terminal
    for cls, prob in zip(le.classes_, pred_proba):
        bar_len  = int(prob * 30)
        bar      = "█" * bar_len + "░" * (30 - bar_len)
        marker   = " ◄ PREDICTED" if cls == pred_label else ""
        print(f"  {cls:20s} {bar} {prob*100:5.1f}%{marker}")

    return pred_label, confidence


def show_model_stats():
    """Show what the model was trained on."""
    import json
    print("\n  📊 Model Training Statistics:")
    for obj in ["informativeness", "damage"]:
        path = os.path.join(RESULTS_DIR, f"metrics_{obj}.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            label = "Informativeness" if obj == "informativeness" else "Damage Severity"
            print(f"\n     {label}:")
            print(f"       Training images : {int(m.get('train_images', 0)):,}")
            print(f"       Test images     : {int(m.get('test_images', 0)):,}")
            print(f"       Accuracy        : {m.get('accuracy', 0)*100:.2f}%")
            print(f"       F1 Score        : {m.get('f1', 0)*100:.2f}%")


# ══════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════
if __name__ == "__main__":

    print("""
╔═══════════════════════════════════════════════╗
║        DisasterRes-Net  —  LIVE DEMO          ║
║   Social Media Image Classification System    ║
╚═══════════════════════════════════════════════╝""")

    # ── Get image path ──
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"\n❌ Image not found: {image_path}")
            sys.exit(1)
    else:
        image_path = find_sample_image()
        if not image_path:
            print("\n❌ No test images found in split_dataset/")
            print("   Usage: python demo.py path/to/image.jpg")
            sys.exit(1)
        print(f"\n  ℹ️  No image specified — using sample from dataset")

    # ── Show model stats first ──
    banner("STEP 1 — Trained Model Statistics")
    show_model_stats()

    # ── Show image info ──
    banner("STEP 2 — Input Image")
    load_and_show_image(image_path)

    # ── Load extractor ──
    banner("STEP 3 — Extracting Features")
    if not TF_OK:
        print("  ❌ TensorFlow not available.")
        sys.exit(1)

    extractor = build_extractor()
    print(f"  ✅ Deep features  : 1000-dimensional (InceptionResNetV2)")
    print(f"  ✅ GLCM features  : 20-dimensional (4 angles × 5 stats)")
    print(f"  ✅ Fused vector   : 1020-dimensional")

    # ── Run predictions ──
    banner("STEP 4 — Predictions")

    predict(image_path, "informativeness", extractor)
    predict(image_path, "damage",          extractor)

    # ── Final summary ──
    banner("DEMO COMPLETE ✅", "═")
    print("""
  This system:
  • Filters useful disaster images (Objective 1)
  • Classifies damage severity (Objective 2)
  • Can process ~1 image/second in real-time
  • Trained on 18,011 CrisisMMD social media images
    """)
