import warnings
warnings.filterwarnings("ignore")

import os
import sys
import joblib
import numpy as np
from PIL import Image

# ── PyTorch + timm ──
import torch
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Import feature functions ──
# Ensure step3_classify.py is in the same directory
try:
    from step3_classify import (
        build_feature_extractor,
        extract_glcm_features,
        extract_lbp_features,
        extract_hsv_features
    )
except ImportError:
    print("❌ Error: Could not find 'step3_classify.py'. Ensure it is in the script directory.")
    sys.exit(1)

IMG_SIZE    = (299, 299)
MODEL_DIR   = "saved_models"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def banner(text):
    print("\n" + "═" * 55)
    print(f"  {text}")
    print("═" * 55)

def load_and_show_info(image_path):
    img = Image.open(image_path).convert("RGB")
    print(f"\n📸 Image     : {os.path.basename(image_path)}")
    print(f"📁 Path      : {image_path}")
    print(f"📐 Size      : {img.width} × {img.height}")
    return img

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

def predict(image_path, objective, extractor):
    pipe_path = os.path.join(MODEL_DIR, f"pipeline_{objective}.joblib")
    le_path   = os.path.join(MODEL_DIR, f"le_{objective}.joblib")

    if not os.path.exists(pipe_path) or not os.path.exists(le_path):
        print(f"❌ Model or LabelEncoder not found for: {objective}")
        return None, None

    pipe = joblib.load(pipe_path)
    le   = joblib.load(le_path)

    # ---- Deep features ----
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    
    # Convert to Tensor (CHW) and cast to Float32
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        deep_feat = extractor(tensor).cpu().numpy()

    # ---- Handcrafted features ----
    # Ensure these are reshaped to (1, -1) for hstack
    glcm_feat = extract_glcm_features(image_path).reshape(1, -1)
    lbp_feat  = extract_lbp_features(image_path).reshape(1, -1)
    hsv_feat  = extract_hsv_features(image_path).reshape(1, -1)

    # Combine all features into one row
    X = np.hstack([deep_feat, glcm_feat, lbp_feat, hsv_feat])

    # ---- Prediction ----
    probs = pipe.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx] * 100

    print(f"🎯 {objective.upper().replace('_', ' ')}")
    print(f"   Prediction : {pred_label}")
    print(f"   Confidence : {confidence:.2f}%")

    return pred_label, confidence

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║        DisasterRes-Net  —  LIVE DEMO          ║
    ╚═══════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        image_input = sys.argv[1]
        if not os.path.exists(image_input):
            print(f"❌ Image not found: {image_input}")
            sys.exit(1)
    else:
        print("Usage: python demo.py <path_to_image>")
        sys.exit(1)

    banner("INPUT IMAGE")
    load_and_show_info(image_input)

    banner("LOADING MODEL")
    # Move extractor to device immediately
    model_extractor = build_feature_extractor().to(device)
    model_extractor.eval()

    banner("PREDICTIONS")
    predict(image_input, "disaster_type", model_extractor)
    predict(image_input, "damage", model_extractor)

    banner("DONE ✅")