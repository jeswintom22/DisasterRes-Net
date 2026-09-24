from __future__ import annotations
import numpy as np
from crew.pipeline_singleton import INFERENCE_POOL, get_damage_analyzer, get_pipeline

def analyze_image(image: np.ndarray, disaster_label: str):
    def work():
        analysis = get_pipeline().analyze(image)
        damage = analysis["predictions"].get("damage", {})
        assessment = get_damage_analyzer().assess(image, analysis["saliency"].saliency_map, disaster_label, localization_backend="provided", lbp_texture_map=analysis["lbp"].texture_map, lbp_statistics=analysis["lbp"].statistics, damage_prediction=damage.get("prediction"), damage_confidence=damage.get("confidence"))
        return analysis, assessment
    return INFERENCE_POOL.submit(work).result()

class DisasterResPipelineTool:
    def run(self, image: np.ndarray): return get_pipeline().analyze(image)

class DamageAssessmentTool:
    def run(self, image: np.ndarray, saliency_map: np.ndarray, disaster_label: str): return get_damage_analyzer().assess(image, saliency_map, disaster_label, localization_backend="provided")
