import warnings
warnings.filterwarnings("ignore")

import os
import sys

# ── Import updated prediction function ──
try:
    from step3_classify import predict_single_image
except ImportError as e:
    print(f"❌ Error importing from step3_classify.py: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def banner(text):
    print("\n" + "═" * 55)
    print(f"  {text}")
    print("═" * 55)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
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
    print(f"\n📸 Image     : {os.path.basename(image_input)}")
    print(f"📁 Path      : {image_input}")

    banner("LOADING MODEL & PREDICTING")
    # All inference and PyTorch logic is internally handled by the rewritten step3_classify.py
    results = predict_single_image(image_input)
    
    if "error" in results:
        print(f"❌ Error during inference: {results['error']}")
        sys.exit(1)

    for objective in ("disaster_type", "damage"):
        if objective in results:
            pred_label = results[objective]
            if pred_label == "model_not_trained":
                print(f"🎯 {objective.upper().replace('_', ' ')}")
                print("   ❌ Model not found (Train step3_classify.py first)")
            else:
                confidence_str = results.get(f"{objective}_confidence", "0.0")
                confidence = float(confidence_str) * 100
                print(f"🎯 {objective.upper().replace('_', ' ')}")
                print(f"   Prediction : {pred_label}")
                print(f"   Confidence : {confidence:.2f}%")

    banner("DONE ✅")

if __name__ == "__main__":
    main()