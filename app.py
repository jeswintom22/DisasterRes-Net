import io

import csv
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from agents.cache import AgentCache
from agents.image_scrape_agent import ImageScrapeAgent
from agents.image_validator_agent import ImageValidatorAgent
from agents.schemas import DisasterEvent
from config.agent_config import load_twikit_credentials
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

DISASTER_LABELS = ("earthquake", "flood", "hurricane", "wildfire", "landslide")
RAW_DATASET_DIR = Path("raw_dataset")
PROVENANCE_PATH = RAW_DATASET_DIR / "step0_provenance.csv"
LIVE_EVENTS_PATH = Path("results") / "live_disaster_events.json"
LIVE_ASSESSMENTS_PATH = Path("results") / "live_image_assessments.json"
_monitor_lock = threading.Lock()
_monitor_active = False
_assessment_lock = threading.Lock()
_incident_jobs: dict[str, dict] = {}
_incident_jobs_lock = threading.Lock()


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


def _analyze_rgb(original_rgb: np.ndarray, disaster_label: str, localization_backend: str = "provided") -> dict:
    """Use the same trained hybrid pipeline for uploads and live agent imagery."""
    analysis = get_pipeline().analyze(original_rgb)
    predictions = analysis["predictions"]
    informativeness_prediction = predictions.get("informativeness") or next(iter(predictions.values()))
    damage_prediction = predictions.get("damage", {})
    assessment = _damage_analyzer.assess(
        original_rgb, analysis["saliency"].saliency_map, disaster_label,
        localization_backend=localization_backend,
        lbp_texture_map=analysis["lbp"].texture_map,
        lbp_statistics=analysis["lbp"].statistics,
        damage_prediction=damage_prediction.get("prediction"),
        damage_confidence=damage_prediction.get("confidence"),
    )
    saliency_map = analysis["saliency"].saliency_map
    lbp_texture = analysis["lbp"].texture_map
    mask_rgb = np.repeat((assessment.damage_mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    saliency_overlay = heatmap_overlay(original_rgb, saliency_map, alpha=0.48)
    return {
        "disaster_type": _prediction_payload(informativeness_prediction),
        "informativeness": _prediction_payload(predictions.get("informativeness", informativeness_prediction)),
        "damage": _prediction_payload(predictions.get("damage", informativeness_prediction)),
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
        "disaster_description": DISASTER_DESCRIPTIONS.get(disaster_label, DISASTER_DESCRIPTIONS["informative"]),
    }


def _read_json(path: Path, fallback: dict) -> dict:
    try:
        with path.open(encoding="utf-8") as fobj:
            return json.load(fobj)
    except (OSError, ValueError):
        return fallback


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as fobj:
        json.dump(payload, fobj, indent=2)
    os.replace(temporary, path)


def _provenance_rows() -> list[dict]:
    if not PROVENANCE_PATH.exists():
        return []
    try:
        with PROVENANCE_PATH.open(newline="", encoding="utf-8") as fobj:
            return list(csv.DictReader(fobj))
    except OSError:
        return []


def _live_assessments() -> list[dict]:
    """Classify new validated agent images and cache the shared model output."""
    with _assessment_lock:
        payload = _read_json(LIVE_ASSESSMENTS_PATH, {"items": []})
        existing = {item["filename"]: item for item in payload.get("items", [])}
        for row in _provenance_rows():
            filename = row.get("filename", "")
            image_path = RAW_DATASET_DIR / row.get("disaster_class", "") / filename
            if not filename or filename in existing or not image_path.is_file():
                continue
            try:
                with Image.open(image_path) as image:
                    result = _analyze_rgb(np.array(image.convert("RGB"), dtype=np.uint8), row["disaster_class"])
                damage = result["damage"]
                assessment = result["damage_assessment"]
                damage_label = damage.get("prediction", "unknown")
                severity = assessment.get("severity_level", "unknown")
                existing[filename] = {
                    "filename": filename,
                    "image_url": f"/api/live/image/{row['disaster_class']}/{filename}",
                    "event_id": row.get("event_id"), "event_source": row.get("event_source"),
                    "disaster_class": row.get("disaster_class"),
                    "place_name": row.get("place_name") or "Location pending",
                    "latitude": row.get("latitude"), "longitude": row.get("longitude"),
                    "tweet_created_at": row.get("tweet_created_at"), "collected_at": row.get("collected_at"),
                    "damage_class": damage_label, "damage_confidence": damage.get("confidence", 0),
                    "dem_score": assessment.get("dem_score", 0), "severity_level": severity,
                    "affected_area_percentage": assessment.get("affected_area_percentage", 0),
                    "urgent": damage_label in {"severe", "critical"} or severity in {"severe", "critical"},
                    "analysis": result,
                }
            except Exception as exc:
                print(f"[LIVE] Could not classify {image_path}: {exc}")
        items = sorted(existing.values(), key=lambda item: item.get("collected_at", ""), reverse=True)
        _write_json(LIVE_ASSESSMENTS_PATH, {"updated_at": datetime.now(timezone.utc).isoformat(), "items": items})
        return items


def _event_by_id(event_id: str) -> dict | None:
    events = _read_json(LIVE_EVENTS_PATH, {"events": []}).get("events", [])
    return next((event for event in events if event.get("event_id") == event_id), None)


def _event_from_payload(payload: dict) -> DisasterEvent:
    event_time = datetime.fromisoformat(str(payload["event_time"]).replace("Z", "+00:00")).replace(tzinfo=None)
    return DisasterEvent(
        event_id=str(payload["event_id"]), source=str(payload["source"]),
        disaster_class=str(payload["disaster_class"]), latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]), place_name=payload.get("place_name"),
        magnitude_or_severity=payload.get("magnitude_or_severity"), event_time=event_time,
        title=str(payload.get("title") or payload["event_id"]),
    )


def _event_assessments(event_id: str) -> list[dict]:
    return [item for item in _live_assessments() if item.get("event_id") == event_id]


def _update_job(event_id: str, **changes: object) -> None:
    with _incident_jobs_lock:
        job = _incident_jobs.get(event_id)
        if job is not None:
            job.update(changes)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()


def _run_incident_analysis(event_payload: dict) -> None:
    event_id = str(event_payload["event_id"])
    cache = AgentCache()
    try:
        event = _event_from_payload(event_payload)
        _update_job(event_id, status="locating", message="Resolving incident location", stage=1)
        if load_twikit_credentials() is None:
            _update_job(event_id, status="blocked", message="Social image search needs Twikit credentials in .env", stage=1)
            return
        if not cache.can_search_today():
            _update_job(event_id, status="blocked", message="Daily social search cap reached", stage=2)
            return
        _update_job(event_id, status="searching", message="Searching incident-specific social imagery", stage=2)
        candidates = ImageScrapeAgent(cache).collect_for_event(event)
        _update_job(event_id, candidate_count=len(candidates), status="validating", message="Validating image evidence", stage=3)
        saved = ImageValidatorAgent().process(candidates)
        _update_job(event_id, saved_count=saved, status="classifying", message="Running hybrid damage classification", stage=4)
        assessments = _event_assessments(event_id)
        if assessments:
            _update_job(event_id, status="complete", message=f"{len(assessments)} image assessment(s) ready", stage=5)
        else:
            _update_job(event_id, status="no_imagery", message="No valid incident imagery was found in this search", stage=5)
    except Exception as exc:
        _update_job(event_id, status="failed", message=f"Analysis could not complete: {exc}", stage=5)
    finally:
        cache.close()


def _resolve_disaster_context(form, filename: str | None) -> dict:
    candidates = [
        ("form", form.get("disaster_type") or form.get("disaster_label") or form.get("disaster_context")),
        ("filename", filename or ""),
    ]
    for source, value in candidates:
        text = str(value or "").lower()
        for label in DISASTER_LABELS:
            if label in text:
                return {"label": label, "source": source}
    return {"label": "unknown", "source": "fallback"}


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
        informativeness_prediction = predictions.get("informativeness") or next(iter(predictions.values()))
        damage_prediction = predictions.get("damage", {})
        disaster_context = _resolve_disaster_context(request.form, file.filename)
        disaster_label = disaster_context["label"]
        localization_backend = request.form.get("localization_backend", "provided")
        assessment = _damage_analyzer.assess(
            original_rgb,
            analysis["saliency"].saliency_map,
            disaster_label,
            localization_backend=localization_backend,
            lbp_texture_map=analysis["lbp"].texture_map,
            lbp_statistics=analysis["lbp"].statistics,
            damage_prediction=damage_prediction.get("prediction"),
            damage_confidence=damage_prediction.get("confidence"),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    saliency_map = analysis["saliency"].saliency_map
    lbp_texture = analysis["lbp"].texture_map
    mask_rgb = np.repeat((assessment.damage_mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    saliency_overlay = heatmap_overlay(original_rgb, saliency_map, alpha=0.48)

    result = {
        "disaster_type": _prediction_payload(informativeness_prediction),
        "informativeness": _prediction_payload(predictions.get("informativeness", informativeness_prediction)),
        "damage": _prediction_payload(predictions.get("damage", informativeness_prediction)),
        "disaster_context": disaster_context,
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


@app.route("/api/live/incidents", methods=["GET"])
def live_incidents():
    events_payload = _read_json(LIVE_EVENTS_PATH, {"updated_at": None, "events": []})
    assessments = _live_assessments()
    return jsonify({
        "updated_at": events_payload.get("updated_at"),
        "monitor_active": _monitor_active,
        "events": events_payload.get("events", [])[:60],
        "assessments": assessments[:24],
        "summary": {
            "active_events": len(events_payload.get("events", [])),
            "images_assessed": len(assessments),
            "urgent_damage": sum(1 for item in assessments if item.get("urgent")),
        },
    })


@app.route("/api/live/incidents/<event_id>/analyze", methods=["POST"])
def analyze_incident(event_id: str):
    event_payload = _event_by_id(event_id)
    if event_payload is None:
        return jsonify({"error": "Incident was not found in the current live feed."}), 404

    cached = _event_assessments(event_id)
    with _incident_jobs_lock:
        job = _incident_jobs.get(event_id)
        if job and job["status"] in {"locating", "searching", "validating", "classifying"}:
            return jsonify({"job": job, "assessments": cached}), 202
        if cached:
            completed = {
                "event_id": event_id, "status": "complete", "stage": 5,
                "message": f"{len(cached)} cached image assessment(s) ready",
                "candidate_count": 0, "saved_count": len(cached),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _incident_jobs[event_id] = completed
            return jsonify({"job": completed, "assessments": cached})
        job = {
            "event_id": event_id, "status": "locating", "stage": 1,
            "message": "Preparing incident analysis", "candidate_count": 0, "saved_count": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _incident_jobs[event_id] = job
    threading.Thread(
        target=_run_incident_analysis, args=(event_payload,), name=f"incident-{event_id}", daemon=True
    ).start()
    return jsonify({"job": job, "assessments": cached}), 202


@app.route("/api/live/incidents/<event_id>/analysis", methods=["GET"])
def incident_analysis(event_id: str):
    if _event_by_id(event_id) is None:
        return jsonify({"error": "Incident was not found in the current live feed."}), 404
    assessments = _event_assessments(event_id)
    with _incident_jobs_lock:
        job = _incident_jobs.get(event_id)
    if job is None:
        job = {
            "event_id": event_id,
            "status": "complete" if assessments else "idle",
            "stage": 5 if assessments else 0,
            "message": f"{len(assessments)} cached image assessment(s) ready" if assessments else "Select this incident to begin analysis",
            "candidate_count": 0,
            "saved_count": len(assessments),
        }
    return jsonify({"job": job, "assessments": assessments})


@app.route("/api/live/image/<disaster_class>/<filename>", methods=["GET"])
def live_image(disaster_class: str, filename: str):
    if disaster_class not in DISASTER_LABELS or Path(filename).name != filename:
        return jsonify({"error": "Invalid image path"}), 400
    return send_from_directory(RAW_DATASET_DIR / disaster_class, filename)


def _run_monitor() -> None:
    global _monitor_active
    try:
        from agents.coordinator_agent import CoordinatorAgent
        CoordinatorAgent().run()
    finally:
        _monitor_active = False
        _monitor_lock.release()


@app.route("/api/live/refresh", methods=["POST"])
def refresh_live_monitor():
    global _monitor_active
    if not _monitor_lock.acquire(blocking=False):
        return jsonify({"started": False, "message": "A live monitoring pass is already running."}), 202
    _monitor_active = True
    threading.Thread(target=_run_monitor, name="disaster-monitor", daemon=True).start()
    return jsonify({"started": True, "message": "Live monitoring started."}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("DISASTERRES_PORT", "5000")))
