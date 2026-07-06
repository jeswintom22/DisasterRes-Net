"""Live single-image demo for the PyTorch-backed DisasterRes-Net pipeline."""

import json
import os
import sys
import warnings

from PIL import Image

from step3_classify import predict_image

warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
SAMPLE_DIRS = [
    os.path.join("split_dataset", "test", "severe"),
    os.path.join("split_dataset", "test", "informative"),
    os.path.join("split_dataset", "test", "earthquake"),
    os.path.join("split_dataset", "test", "flood"),
]


def banner(text, char="="):
    width = 55
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)


def find_sample_image():
    for directory in SAMPLE_DIRS:
        if os.path.exists(directory):
            images = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if images:
                return os.path.join(directory, images[0])
    for root, _, files in os.walk("split_dataset"):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(root, name)
    return None


def show_model_stats():
    print("\n  Model Training Statistics:")
    for objective in ["informativeness", "damage"]:
        path = os.path.join(RESULTS_DIR, f"metrics_{objective}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fobj:
            metrics = json.load(fobj)
        label = "Informativeness" if objective == "informativeness" else "Damage Severity"
        print(f"\n    {label}:")
        print(f"      Training images : {int(metrics.get('train_images', 0)):,}")
        print(f"      Test images     : {int(metrics.get('test_images', 0)):,}")
        print(f"      Accuracy        : {metrics.get('accuracy', 0) * 100:.2f}%")
        print(f"      F1 Score        : {metrics.get('f1', 0) * 100:.2f}%")


def describe_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        print(f"\n  Image : {os.path.basename(image_path)}")
        print(f"  Path  : {image_path}")
        print(f"  Size  : {img.width} x {img.height} px")


def run_prediction(image_path, objective):
    result = predict_image(image_path, objective)
    title = "Informativeness" if objective == "informativeness" else "Damage Severity"
    print(f"\n  {'-' * 48}")
    print(f"  {title}")
    print(f"  {'-' * 48}")
    print(f"  Prediction : {result['prediction'].upper().replace('_', ' ')}")
    if "confidence" in result:
        print(f"  Confidence : {result['confidence'] * 100:.1f}%")

    probabilities = result.get("probabilities") or {}
    for cls_name, prob in probabilities.items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        marker = " <- predicted" if cls_name == result["prediction"] else ""
        print(f"  {cls_name:20s} {bar} {prob * 100:5.1f}%{marker}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            raise SystemExit(1)
    else:
        image_path = find_sample_image()
        if not image_path:
            print("No test images found in split_dataset/")
            raise SystemExit(1)

    banner("STEP 1 - Trained Model Statistics")
    show_model_stats()

    banner("STEP 2 - Input Image")
    describe_image(image_path)

    banner("STEP 3 - Predictions")
    run_prediction(image_path, "informativeness")
    run_prediction(image_path, "damage")

    banner("DEMO COMPLETE")
    print("\n  This demo uses the saved PyTorch feature extractor path plus the trained fused-feature classifier.\n")
