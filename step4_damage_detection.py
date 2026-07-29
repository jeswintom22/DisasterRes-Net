"""Step 4: M2 Damage Localization and Damage Assessment.

Pipeline:
  disaster prediction -> saliency detection -> damage segmentation -> DDM -> DEM

The module converts saliency into a binary damage mask, removes noise with
morphology, detects connected damaged regions, and computes a normalized Damage
Evaluation Metric (DEM) from damaged pixel ratio, saliency density, affected
region count, compactness, and disaster severity weighting.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from damage_assessment.localization import DamageLocalizationAnalyzer
from preprocessing.saliency import SaliencyAttention
from utils.image_io import image_to_data_url

_SPLIT_DIR = os.path.normpath(os.path.join("split_dataset"))
_RESULT_DIR = os.path.normpath(os.path.join("results"))
_DDM_DIR = os.path.normpath(os.path.join("results", "ddm_heatmaps"))
_IMG_EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(_RESULT_DIR, exist_ok=True)
os.makedirs(_DDM_DIR, exist_ok=True)

_saliency = SaliencyAttention()
_analyzer = DamageLocalizationAnalyzer()


def compute_saliency_map(img_rgb: np.ndarray) -> np.ndarray:
    """Return the saliency map used by attention and M2 localization."""
    return _saliency.process(img_rgb).saliency_map


def compute_dem_score(saliency_map: np.ndarray) -> float:
    """Compatibility DEM proxy for older callers; full DEM is in process_image."""
    return float(np.clip(np.mean(saliency_map) * 100.0, 0.0, 100.0))


def dem_to_damage_class(dem: float) -> str:
    """Map a 0-100 DEM score to severity level."""
    if dem <= 20:
        return "minimal"
    if dem <= 40:
        return "mild"
    if dem <= 60:
        return "moderate"
    if dem <= 80:
        return "severe"
    return "critical"


def _save_image_data_url(data_url: str, output_path: str) -> None:
    import base64

    _, encoded = data_url.split(",", 1)
    with open(output_path, "wb") as fobj:
        fobj.write(base64.b64decode(encoded))


def process_image(image_path: str, disaster_label: str = "unknown") -> Optional[Dict[str, object]]:
    """Run complete M2 analysis for one image."""
    try:
        with Image.open(image_path) as img:
            img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        print(f"[WARN] Could not process '{image_path}': {exc}")
        return None

    saliency = compute_saliency_map(img_rgb)
    assessment = _analyzer.assess(img_rgb, saliency, disaster_label)
    return {
        "dem": assessment.dem_score,
        "damage_class": assessment.severity_level,
        "affected_area_percentage": assessment.affected_area_percentage,
        "damaged_region_count": assessment.damaged_region_count,
        "largest_region_percentage": assessment.largest_region_percentage,
        "average_damage_density": assessment.average_damage_density,
        "localization_score": assessment.localization_score,
        "centroid": assessment.centroid,
        "boundaries": assessment.boundaries,
        "regions": [asdict(region) for region in assessment.regions],
        "impact_estimates": assessment.impact_estimates,
        "emergency_recommendations": assessment.emergency_recommendations,
        "ddm_image": image_to_data_url(assessment.ddm_overlay),
    }


def process_dataset_split(split: str = "test") -> Dict[str, Dict[str, object]]:
    """Process every image in a split and save DDM visualizations."""
    split_dir = os.path.normpath(os.path.join(_SPLIT_DIR, split))
    if not os.path.isdir(split_dir):
        print(f"[ERROR] Split folder not found: {split_dir}")
        return {}

    print(f"\n[STEP] Processing {split.upper()} split for M2 localization")
    results: Dict[str, Dict[str, object]] = {}

    for class_folder in sorted(os.listdir(split_dir)):
        class_path = os.path.normpath(os.path.join(split_dir, class_folder))
        if not os.path.isdir(class_path):
            continue
        image_files = [name for name in os.listdir(class_path) if name.lower().endswith(_IMG_EXTS)]
        print(f"  Processing class '{class_folder}' ({len(image_files)} images)...")

        for image_file in tqdm(image_files, desc=f"    {class_folder}"):
            image_path = os.path.normpath(os.path.join(class_path, image_file))
            proc = process_image(image_path, disaster_label=class_folder)
            if proc is None:
                continue
            rel_path = os.path.relpath(image_path, _SPLIT_DIR)
            heatmap_name = rel_path.replace(os.sep, "_").replace(".", "_") + "_ddm.jpg"
            heatmap_path = os.path.normpath(os.path.join(_DDM_DIR, heatmap_name))
            _save_image_data_url(str(proc.pop("ddm_image")), heatmap_path)
            proc["heatmap_path"] = heatmap_path
            results[image_path] = proc
    return results


def compute_m2_metrics(results_dict: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    """Aggregate M2 metrics for a processed split."""
    if not results_dict:
        return {}
    dems = np.asarray([float(item["dem"]) for item in results_dict.values()], dtype=np.float32)
    affected = np.asarray([float(item["affected_area_percentage"]) for item in results_dict.values()], dtype=np.float32)
    regions = np.asarray([float(item["damaged_region_count"]) for item in results_dict.values()], dtype=np.float32)
    return {
        "total_images": float(len(results_dict)),
        "dem_mean": float(dems.mean()),
        "dem_std": float(dems.std()),
        "dem_min": float(dems.min()),
        "dem_max": float(dems.max()),
        "affected_area_mean": float(affected.mean()),
        "damaged_region_count_mean": float(regions.mean()),
    }


def save_m2_report(split_results: Dict[str, Dict[str, object]], split: str = "test") -> str:
    """Save a JSON report with DEM/DDM metrics and region statistics."""
    report = {
        "module": "M2",
        "split": split,
        "metrics": compute_m2_metrics(split_results),
        "dem_scale": "0-20 Minimal, 21-40 Mild, 41-60 Moderate, 61-80 Severe, 81-100 Critical",
        "pipeline": "saliency -> binary mask -> morphology -> connected components -> DDM -> DEM",
        "results": split_results,
    }
    report_path = os.path.normpath(os.path.join(_RESULT_DIR, f"m2_report_{split}.json"))
    with open(report_path, "w", encoding="utf-8") as fobj:
        json.dump(report, fobj, indent=2)
    print(f"[INFO] M2 report saved: {report_path}")
    return report_path


def main() -> int:
    print("\n" + "=" * 70)
    print("STEP 4 - M2 Damage Localization, DDM, and DEM")
    print("=" * 70)
    test_results = process_dataset_split(split="test")
    if test_results:
        save_m2_report(test_results, split="test")
        print(f"[RESULT] Processed {len(test_results)} test images")
        print(f"[INFO] DDM maps saved to: {_DDM_DIR}")
    else:
        print("[WARN] No test images processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
