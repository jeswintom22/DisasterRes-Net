import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

from damage_assessment.localization import DamageLocalizationAnalyzer
from models.hybrid_pipeline import HybridDisasterPipeline
from utils.image_io import heatmap_overlay, image_to_data_url

app = Flask(__name__)

_pipeline: HybridDisasterPipeline | None = None
_damage_analyzer = DamageLocalizationAnalyzer()


DISASTER_DESCRIPTIONS = {
    "earthquake": "Seismic impact pattern with likely structural disruption and access constraints.",
    "flood": "Hydrological disaster pattern with possible inundation and infrastructure isolation.",
    "hurricane": "Storm-impact pattern with wind, flood, and debris-related risk zones.",
    "wildfire": "Fire-impact pattern with vegetation loss, heat damage, and spread corridors.",
    "landslide": "Slope-failure pattern with terrain displacement and blocked access risk.",
    "informative": "Image contains disaster-relevant visual evidence for downstream assessment.",
    "not_informative": "Image has limited disaster evidence; damage metrics should be reviewed cautiously.",
}


def get_pipeline() -> HybridDisasterPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridDisasterPipeline()
    return _pipeline


def _percent(value: float) -> float:
    return round(float(value) * 100.0, 2)


def _prediction_payload(prediction: dict) -> dict:
    probabilities = prediction.get("probabilities", {}) or {}
    top5 = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "prediction": prediction.get("prediction", "unknown"),
        "confidence": _percent(prediction.get("confidence", 0.0)),
        "probabilities": {key: _percent(value) for key, value in probabilities.items()},
        "top5": [{"label": key, "confidence": _percent(value)} for key, value in top5],
        "feature_dim": prediction.get("feature_dim", 0),
        "fusion": prediction.get("fusion", {}),
    }


def _regions_payload(regions) -> list[dict]:
    return [
        {
            "area_pixels": region.area_pixels,
            "area_percentage": round(region.area_percentage, 2),
            "centroid": [round(region.centroid[0], 1), round(region.centroid[1], 1)],
            "bbox": list(region.bbox),
            "compactness": round(region.compactness, 4),
            "mean_saliency": round(region.mean_saliency, 4),
            "region_confidence": round(region.region_confidence, 4),
            "rank": region.rank,
            "polygon": [list(point) for point in region.polygon],
        }
        for region in regions[:12]
    ]


def _damage_payload(assessment) -> dict:
    return {
        "dem_score": round(assessment.dem_score, 2),
        "severity_level": assessment.severity_level,
        "affected_area_percentage": round(assessment.affected_area_percentage, 2),
        "damaged_region_count": assessment.damaged_region_count,
        "largest_region_percentage": round(assessment.largest_region_percentage, 2),
        "average_region_size_percentage": round(assessment.average_region_size_percentage, 2),
        "average_damage_density": round(assessment.average_damage_density, 4),
        "boundary_complexity": round(assessment.boundary_complexity, 4),
        "texture_entropy": round(assessment.texture_entropy, 4),
        "localization_confidence": round(assessment.localization_confidence, 4),
        "average_activation": round(assessment.average_activation, 4),
        "localization_score": round(assessment.localization_score, 2),
        "localization_backend": assessment.localization_backend,
        "backend_metadata": assessment.backend_metadata,
        "centroid": None if assessment.centroid is None else [round(assessment.centroid[0], 1), round(assessment.centroid[1], 1)],
        "boundaries": [list(item) for item in assessment.boundaries],
        "regions": _regions_payload(assessment.regions),
        "impact_estimates": {key: round(value, 2) for key, value in assessment.impact_estimates.items()},
        "emergency_recommendations": assessment.emergency_recommendations,
    }


def _texture_stats_payload(stats: dict) -> dict:
    return {key: round(float(value), 4) for key, value in stats.items()}


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

    try:
        analysis = get_pipeline().analyze(original_rgb)
        predictions = analysis["predictions"]
        disaster_prediction = predictions.get("informativeness") or next(iter(predictions.values()))
        disaster_label = disaster_prediction.get("prediction", "unknown")
        localization_backend = request.form.get("localization_backend", "sun_ica")
        assessment = _damage_analyzer.assess(
            original_rgb,
            analysis["saliency"].saliency_map,
            disaster_label,
            localization_backend=localization_backend,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    saliency_map = analysis["saliency"].saliency_map
    lbp_texture = analysis["lbp"].texture_map
    mask_rgb = np.repeat((assessment.damage_mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    saliency_overlay = heatmap_overlay(original_rgb, saliency_map, alpha=0.48)

    result = {
        "disaster_type": _prediction_payload(disaster_prediction),
        "informativeness": _prediction_payload(predictions.get("informativeness", disaster_prediction)),
        "damage": _prediction_payload(predictions.get("damage", disaster_prediction)),
        "damage_assessment": _damage_payload(assessment),
        "visualizations": {
            "original": image_to_data_url(original_rgb),
            "saliency": image_to_data_url((saliency_map * 255).astype(np.uint8), fmt="PNG"),
            "saliency_overlay": image_to_data_url(saliency_overlay),
            "lbp": image_to_data_url((lbp_texture * 255).astype(np.uint8), fmt="PNG"),
            "damage_mask": image_to_data_url(mask_rgb, fmt="PNG"),
            "ddm": image_to_data_url(assessment.ddm_overlay),
            "gradcam": image_to_data_url(saliency_overlay),
            "enhanced": image_to_data_url(analysis["saliency"].enhanced_rgb),
        },
        "texture_statistics": _texture_stats_payload(analysis["lbp"].statistics),
        "processing_time_ms": round(float(analysis["processing_time_ms"]), 2),
        "model_info": analysis["model_info"],
        "prediction_breakdown": {
            "cnn_stream": "Saliency-attended RGB image passed through InceptionResNetV2.",
            "texture_stream": "LBP histogram/statistics describe local rubble, vegetation, water, and burn textures.",
            "fusion": "Feature vectors are concatenated and standardized before the classifier head.",
            "damage_module": "M2 converts saliency into a filtered connected-component damage mask, DDM, and DEM.",
        },
        "disaster_description": DISASTER_DESCRIPTIONS.get(disaster_label, DISASTER_DESCRIPTIONS.get("informative")),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
