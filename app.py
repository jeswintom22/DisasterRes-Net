import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

from step3_classify import predict_image
from step4_damage_detection import compute_saliency_map

app = Flask(__name__)

OBJECTIVES = ("informativeness", "damage")


def generate_heatmap_overlay(original_img_cv, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (original_img_cv.shape[1], original_img_cv.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_color, alpha, original_img_cv, 1 - alpha, 0)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img_bytes = file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Invalid image format: {exc}"}), 400

    original_rgb = np.array(img_pil, dtype=np.uint8)
    original_cv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    heatmap = compute_saliency_map(original_rgb)
    overlay_bgr = generate_heatmap_overlay(original_cv, heatmap)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay_rgb)

    buffered = io.BytesIO()
    overlay_pil.save(buffered, format="JPEG", quality=85)
    overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    temp_input = os.path.join("results", "_web_input_tmp.jpg")
    os.makedirs("results", exist_ok=True)
    img_pil.save(temp_input)

    results = {}
    try:
        for objective in OBJECTIVES:
            try:
                pred = predict_image(temp_input, objective)
                results[objective] = {
                    "prediction": pred["prediction"],
                    "confidence": f"{pred.get('confidence', 0.0) * 100:.2f}",
                    "probabilities": pred.get("probabilities", {}),
                    "heatmap": f"data:image/jpeg;base64,{overlay_b64}",
                }
            except Exception as exc:
                results[objective] = {"error": str(exc)}
    finally:
        try:
            os.remove(temp_input)
        except OSError:
            pass

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
