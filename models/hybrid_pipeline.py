"""Research-grade hybrid inference orchestration.

This module preserves the saved RandomForest checkpoints while making the
active inference path explicit:

RGB image -> saliency attention -> CNN features
RGB image -> LBP texture descriptors
CNN + handcrafted features -> standardized fusion -> classifier head -> calibration & OOD guard
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from fusion.feature_fusion import fuse_feature_streams
from preprocessing.lbp import LBPFeatureExtractor
from preprocessing.saliency import SaliencyAttention, saliency_to_rgb
from step3_classify import (
    FEATURE_PIPELINE,
    OBJECTIVES,
    _load_calibrated_bundle,
    _load_rf_bundle,
    _load_shared_feature_models,
    extract_features_predictions_from_arrays,
    extract_glcm_features_from_rgb,
)


@dataclass(frozen=True)
class ObjectivePrediction:
    """Prediction output for one classification objective."""

    objective: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    feature_dim: int
    fusion: Dict[str, object]
    effective_prediction: str = ""
    uncertain: bool = False
    top2: List[Dict[str, object]] = field(default_factory=list)
    calibrated_probabilities: Dict[str, float] = field(default_factory=dict)
    ood_flag: bool = False


class HybridDisasterPipeline:
    """Run saliency-attended CNN + LBP fusion inference with probability calibration."""

    def __init__(self) -> None:
        self.saliency = SaliencyAttention()
        self.lbp = LBPFeatureExtractor()

    def analyze(self, img_rgb: np.ndarray) -> Dict[str, object]:
        started = time.perf_counter()
        saliency_output = self.saliency.process(img_rgb)
        lbp_output = self.lbp.extract_from_rgb(img_rgb)

        predictions = self._predict_objectives(
            original_rgb=img_rgb,
            enhanced_rgb=saliency_output.enhanced_rgb,
            saliency_rgb=saliency_to_rgb(saliency_output.saliency_map),
            lbp_histogram=lbp_output.histogram,
            lbp_feature_vector=lbp_output.feature_vector,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        backend_mode = os.getenv("DISASTERRES_MODEL_BACKEND", "calibrated_rf").strip().lower()
        return {
            "predictions": {key: value.__dict__ for key, value in predictions.items()},
            "saliency": saliency_output,
            "lbp": lbp_output,
            "processing_time_ms": elapsed_ms,
            "model_info": {
                "backbone": "InceptionResNetV2 via timm",
                "classifier": "StandardScaler + Regularized RandomForest + OOF Calibration",
                "model_backend": backend_mode,
                "feature_pipeline": FEATURE_PIPELINE,
                "saliency_attention": saliency_output.method,
                "lbp_stream": "uniform LBP histogram + texture statistics used in fused classifier input",
                "checkpoint_compatibility": "Uses calibrated bundle + RF/scaler/encoder artifacts when present.",
            },
        }

    def _predict_objectives(
        self,
        original_rgb: np.ndarray,
        enhanced_rgb: np.ndarray,
        saliency_rgb: np.ndarray,
        lbp_histogram: np.ndarray,
        lbp_feature_vector: np.ndarray,
    ) -> Dict[str, ObjectivePrediction]:
        model, _ = _load_shared_feature_models()
        predictions: Dict[str, ObjectivePrediction] = {}

        cnn_features = extract_features_predictions_from_arrays(
            model,
            [enhanced_rgb, saliency_rgb],
            batch_size=2,
            show_progress=False,
        )
        df1 = cnn_features[:1]
        df2 = cnn_features[1:2]
        glcm = np.asarray([extract_glcm_features_from_rgb(original_rgb)], dtype=np.float32)
        available_features = {
            "cnn_original": df1,
            "cnn_saliency": df2,
            "glcm": glcm,
            "lbp_hist": np.asarray([lbp_histogram], dtype=np.float32),
            "lbp_stats": np.asarray([lbp_feature_vector], dtype=np.float32),
        }

        backend_mode = os.getenv("DISASTERRES_MODEL_BACKEND", "calibrated_rf").strip().lower()
        operating_point = os.getenv("DISASTERRES_OPERATING_POINT", "default").strip()

        for objective in OBJECTIVES:
            clf, le, scaler, feature_spec = _load_rf_bundle(objective)
            fused, metadata = fuse_feature_streams(available_features, feature_spec)
            fused_scaled = scaler.transform(fused)

            pred_idx = int(clf.predict(fused_scaled)[0])
            pred_label = str(le.inverse_transform([pred_idx])[0])
            confidence = 0.0
            probabilities: Dict[str, float] = {}
            effective_pred = pred_label
            uncertain = False
            top2: List[Dict[str, object]] = []
            ood_flag = False

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(fused_scaled)[0]
                cal_bundle = _load_calibrated_bundle(objective, feature_spec.name)
                if cal_bundle is not None and backend_mode != "legacy_rf":
                    cal_out = cal_bundle.evaluate_probabilities(
                        probs,
                        cnn_embedding=df1[0],
                        operating_mode=operating_point,
                    )
                    pred_label = str(cal_out["prediction"])
                    effective_pred = str(cal_out["effective_prediction"])
                    uncertain = bool(cal_out["uncertain"])
                    confidence = float(cal_out["confidence"])
                    probabilities = {str(k): float(v) for k, v in cal_out["probabilities"].items()}
                    top2 = list(cal_out["top2"])
                    ood_flag = bool(cal_out["ood_flag"])
                else:
                    confidence = float(np.max(probs))
                    probabilities = {str(cls): float(prob) for cls, prob in zip(le.classes_, probs)}
                    order = np.argsort(probs)[::-1]
                    top2 = [
                        {"label": str(le.classes_[order[0]]), "confidence": round(float(probs[order[0]]) * 100.0, 2)},
                        {"label": str(le.classes_[order[1]]), "confidence": round(float(probs[order[1]]) * 100.0, 2)},
                    ]

            predictions[objective] = ObjectivePrediction(
                objective=objective,
                prediction=pred_label,
                confidence=confidence,
                probabilities=probabilities,
                feature_dim=int(fused.shape[1]),
                fusion={**metadata.as_dict(), "artifact_pipeline": feature_spec.name},
                effective_prediction=effective_pred,
                uncertain=uncertain,
                top2=top2,
                calibrated_probabilities=probabilities,
                ood_flag=ood_flag,
            )
        return predictions
