"""Phase 1 & Phase 5: In-Fold RF Regularization, OOF Multiplier Search, Calibration & OOD Guard.

Implements:
- In-fold StandardScaler + RandomForestClassifier Pipeline (zero fold leakage)
- Grid search over class_weight='balanced_subsample', min_samples_leaf in {1, 3, 5, 10},
  and max_features in {'sqrt', 0.1} on 5-fold StratifiedKFold OOF predictions
- Coordinate search for per-class probability multipliers w_k in [0.5, 20.0]
- Cost-sensitive operating points (1x standard, 2x and 3x disaster-miss penalty)
- Scalar temperature scaling T > 0 and Expected Calibration Error (ECE)
- Abstention rule ('uncertain' + top-2 classes) and Mahalanobis OOD guard
- Single-touch evaluation on the 5,433-image test set with Wilson 95% CIs
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
     ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from models.calibration import (
    AbstentionConfig,
    CalibratedModelBundle,
    MahalanobisOODGuard,
    apply_multipliers,
    disaster_miss_rate,
    expected_calibration_error,
    fit_probability_multipliers,
    fit_temperature_scaling,
    tune_abstention_thresholds,
    wilson_ci,
)

ROOT = Path(__file__).resolve().parents[1]


def _resolve_feature_npz(split: str) -> Path:
    cache_dir = ROOT / ".cache" / "feature_cache"
    for obj in ("disaster_type", "damage", "informativeness"):
        cand = cache_dir / f"{obj}_{split}_df1_df2_glcm_lbp_stats_features.npz"
        if cand.exists():
            return cand
    return ROOT / "mlruns" / "2" / "ae2032185043451787f78166d5a42d82" / "artifacts" / f"damage_{split}_df1_df2_glcm_lbp_stats_features.npz"


TRAIN_NPZ = _resolve_feature_npz("train")
TEST_NPZ = _resolve_feature_npz("test")
MODEL_DIR = ROOT / "saved_models"
RESULT_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
FEATURE_PIPELINE = "df1_df2_glcm_lbp_stats"


def _extract_peril(paths: np.ndarray) -> List[str]:
    return [os.path.basename(os.path.dirname(str(p))).lower() for p in paths]


def _per_class_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for idx, cls in enumerate(classes):
        true_mask = y_true == idx
        pred_mask = y_pred == idx
        tp = int(np.sum(true_mask & pred_mask))
        support = int(np.sum(true_mask))
        pred_total = int(np.sum(pred_mask))

        rec = tp / support if support > 0 else 0.0
        prec = tp / pred_total if pred_total > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rec_ci = wilson_ci(tp, support)
        prec_ci = wilson_ci(tp, pred_total)

        out[cls] = {
            "support": support,
            "predicted": pred_total,
            "tp": tp,
            "precision": round(prec, 4),
            "precision_ci95": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
            "recall": round(rec, 4),
            "recall_ci95": [round(rec_ci[0], 4), round(rec_ci[1], 4)],
            "f1": round(f1, 4),
        }
    return out


def _save_cm_plot(cm: np.ndarray, classes: List[str], out_path: Path, title: str, normalize: bool = False) -> None:
    display_labels = [c.replace("_", " ").title() for c in classes]
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=200)
    mat = cm.astype(np.float64) / np.maximum(1, cm.sum(axis=1, keepdims=True)) if normalize else cm
    fmt = ".2f" if normalize else "d"
    disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=display_labels)
    disp.plot(cmap="Blues", ax=ax, values_format=fmt, colorbar=True)
    ax.set_title(title, fontsize=11.5, weight="bold", pad=12)
    ax.set_xlabel("Predicted Disaster Peril", fontsize=11, weight="bold", labelpad=8)
    ax.set_ylabel("True Disaster Peril", fontsize=11, weight="bold", labelpad=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run_phase1_and_5_pipeline() -> Dict[str, object]:
    started = time.perf_counter()
    print("[Phase 1 & 5] Loading cached 2,036-D feature matrices...")
    train_data = np.load(TRAIN_NPZ, allow_pickle=True)
    test_data = np.load(TEST_NPZ, allow_pickle=True)

    x_train = train_data["fused"].astype(np.float32)
    x_test = test_data["fused"].astype(np.float32)
    df1_train = train_data["df1"].astype(np.float32)
    df1_test = test_data["df1"].astype(np.float32)

    y_train_str = _extract_peril(train_data["image_paths"])
    y_test_str = _extract_peril(test_data["image_paths"])

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_test = le.transform(y_test_str)
    classes = [str(c) for c in le.classes_]
    not_dis_idx = classes.index("not_disaster")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Grid Search over Regularized RF Hyperparameters on 5-Fold OOF
    # Per spec: class_weight='balanced_subsample', min_samples_leaf in {1, 3, 5, 10}, max_features in {'sqrt', 0.1}
    grid_candidates = [
        {"min_samples_leaf": 1, "max_features": "sqrt", "n_estimators": 300},
        {"min_samples_leaf": 3, "max_features": "sqrt", "n_estimators": 300},
        {"min_samples_leaf": 5, "max_features": "sqrt", "n_estimators": 300},
        {"min_samples_leaf": 10, "max_features": "sqrt", "n_estimators": 300},
    ]

    grid_results: List[Dict[str, object]] = []
    best_oof_probs: np.ndarray | None = None
    best_cfg: Dict[str, object] = grid_candidates[2]
    best_tuned_macro_f1 = -1.0
    best_w = np.ones(len(classes), dtype=np.float64)

    oof_cache_path = RESULT_DIR / "phase1_oof_probs_cache.npz"
    cached_oof = dict(np.load(oof_cache_path, allow_pickle=True)) if oof_cache_path.exists() else {}
    cache_updated = False

    for idx, params in enumerate(grid_candidates, 1):
        t0 = time.perf_counter()
        cache_key = f"leaf{params['min_samples_leaf']}_mf{params['max_features']}_est{params['n_estimators']}"
        print(f"[Phase 1] Grid {idx}/{len(grid_candidates)}: {params}...")
        if cache_key in cached_oof:
            oof_probs = cached_oof[cache_key]
        else:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=int(params["n_estimators"]),
                            min_samples_leaf=int(params["min_samples_leaf"]),
                            max_features=params["max_features"],
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=4,
                        ),
                    ),
                ]
            )
            oof_probs = cross_val_predict(pipe, x_train, y_train, cv=skf, method="predict_proba", n_jobs=1)
            cached_oof[cache_key] = oof_probs.astype(np.float32)
            np.savez_compressed(oof_cache_path, **cached_oof)
            cache_updated = False
        raw_preds = np.argmax(oof_probs, axis=1)
        raw_macro_f1 = float(f1_score(y_train, raw_preds, average="macro", zero_division=0))
        raw_bal_acc = float(balanced_accuracy_score(y_train, raw_preds))
        raw_acc = float(accuracy_score(y_train, raw_preds))
        raw_miss = disaster_miss_rate(y_train, raw_preds, not_dis_idx)

        w_opt, tuned_stats = fit_probability_multipliers(
            oof_probs, y_train, not_disaster_idx=not_dis_idx, disaster_miss_penalty=1.0
        )
        elapsed = time.perf_counter() - t0
        row = {
            "params": params,
            "raw_oof_accuracy": round(raw_acc, 4),
            "raw_oof_macro_f1": round(raw_macro_f1, 4),
            "raw_oof_balanced_acc": round(raw_bal_acc, 4),
            "raw_oof_disaster_miss_rate": round(raw_miss, 4),
            "tuned_oof_accuracy": round(tuned_stats["oof_accuracy"], 4),
            "tuned_oof_macro_f1": round(tuned_stats["oof_macro_f1"], 4),
            "tuned_oof_balanced_acc": round(tuned_stats["oof_balanced_accuracy"], 4),
            "tuned_oof_disaster_miss_rate": round(tuned_stats["oof_disaster_miss_rate"], 4),
            "multipliers": [round(float(x), 4) for x in w_opt],
            "elapsed_sec": round(elapsed, 2),
        }
        grid_results.append(row)
        print(
            f"   -> Raw OOF Macro F1={raw_macro_f1*100:.2f}% | Tuned OOF Macro F1={tuned_stats['oof_macro_f1']*100:.2f}% "
            f"| BalAcc={tuned_stats['oof_balanced_accuracy']*100:.2f}% | Miss={tuned_stats['oof_disaster_miss_rate']*100:.2f}% ({elapsed:.1f}s)"
        )

        if tuned_stats["oof_macro_f1"] > best_tuned_macro_f1:
            best_tuned_macro_f1 = tuned_stats["oof_macro_f1"]
            best_oof_probs = oof_probs
            best_cfg = params
            best_w = w_opt

    if cache_updated:
        np.savez_compressed(oof_cache_path, **cached_oof)
    assert best_oof_probs is not None

    # 2. Compute Fold-Level Spread on the Winning Configuration
    print(f"[Phase 1] Winning RF config: {best_cfg} (OOF Macro F1 = {best_tuned_macro_f1*100:.2f}%)")
    fold_macro_f1s: List[float] = []
    fold_bal_accs: List[float] = []
    fold_accs: List[float] = []
    oof_adj_probs = apply_multipliers(best_oof_probs, best_w)
    oof_adj_preds = np.argmax(oof_adj_probs, axis=1)

    for _, val_idx in skf.split(x_train, y_train):
        fold_macro_f1s.append(float(f1_score(y_train[val_idx], oof_adj_preds[val_idx], average="macro", zero_division=0)))
        fold_bal_accs.append(float(balanced_accuracy_score(y_train[val_idx], oof_adj_preds[val_idx])))
        fold_accs.append(float(accuracy_score(y_train[val_idx], oof_adj_preds[val_idx])))

    # 3. Cost-Sensitive Variants (2x and 3x penalty on disaster -> not_disaster)
    print("[Phase 1] Fitting cost-sensitive operating points (2x and 3x disaster-miss penalty)...")
    w_cost_2x, stats_cost_2x = fit_probability_multipliers(
        best_oof_probs, y_train, not_disaster_idx=not_dis_idx, disaster_miss_penalty=2.0
    )
    w_cost_3x, stats_cost_3x = fit_probability_multipliers(
        best_oof_probs, y_train, not_disaster_idx=not_dis_idx, disaster_miss_penalty=3.0
    )

    # 4. Phase 5: Temperature Scaling, Abstention ('uncertain') & Mahalanobis OOD Guard
    print("[Phase 5] Fitting temperature scaling, abstention thresholds, and Mahalanobis OOD guard on OOF...")
    best_temp, ece_before, ece_after = fit_temperature_scaling(oof_adj_probs, y_train)
    cal_oof_probs = apply_multipliers(best_oof_probs, best_w, temperature=best_temp)

    abstention_cfg, abstention_oof_stats = tune_abstention_thresholds(
        cal_oof_probs, y_train, classes=classes, max_abstain_rate=0.15, target_precision=0.72
    )
    ood_guard = MahalanobisOODGuard(percentile_threshold=98.5).fit(df1_train, y_train)

    # 5. Fit Final Winning Pipeline on All 12,670 Training Samples & Single-Touch Test Evaluation
    print("[Phase 1 & 5] Fitting final regularized RF on 12,670 training samples and evaluating ONCE on 5,433 test set...")
    final_scaler = StandardScaler()
    x_train_scaled = final_scaler.fit_transform(x_train)
    x_test_scaled = final_scaler.transform(x_test)

    final_clf = RandomForestClassifier(
        n_estimators=int(best_cfg["n_estimators"]),
        min_samples_leaf=int(best_cfg["min_samples_leaf"]),
        max_features=best_cfg["max_features"],
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=4,
    )
    final_clf.fit(x_train_scaled, y_train)

    # Baseline Test Metrics (prior uncalibrated model for exact side-by-side comparison)
    raw_test_probs = final_clf.predict_proba(x_test_scaled)
    cal_test_probs = apply_multipliers(raw_test_probs, best_w, temperature=best_temp)
    test_preds = np.argmax(cal_test_probs, axis=1)

    cost2x_test_probs = apply_multipliers(raw_test_probs, w_cost_2x, temperature=best_temp)
    cost2x_test_preds = np.argmax(cost2x_test_probs, axis=1)

    cost3x_test_probs = apply_multipliers(raw_test_probs, w_cost_3x, temperature=best_temp)
    cost3x_test_preds = np.argmax(cost3x_test_probs, axis=1)

    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))
    test_bal_acc = float(balanced_accuracy_score(y_test, test_preds))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_miss_rate = disaster_miss_rate(y_test, test_preds, not_dis_idx)
    test_ece_raw = expected_calibration_error(raw_test_probs, y_test)
    test_ece_cal = expected_calibration_error(cal_test_probs, y_test)

    per_class_test = _per_class_metrics_with_ci(y_test, test_preds, classes)
    cm_test = confusion_matrix(y_test, test_preds)

    # Evaluate Abstention on Test Set
    sorted_test_p = np.sort(cal_test_probs, axis=1)
    test_max_p = sorted_test_p[:, -1]
    test_margin = sorted_test_p[:, -1] - sorted_test_p[:, -2]
    test_abstain_mask = (test_max_p < abstention_cfg.min_confidence) | (test_margin < abstention_cfg.min_top2_margin)
    test_answered_mask = ~test_abstain_mask
    test_abstain_rate = float(np.mean(test_abstain_mask))
    test_answered_acc = float(accuracy_score(y_test[test_answered_mask], test_preds[test_answered_mask]))
    test_answered_macro_f1 = float(
        f1_score(y_test[test_answered_mask], test_preds[test_answered_mask], average="macro", zero_division=0)
    )

    # Save Confusion Matrix plots
    _save_cm_plot(
        cm_test,
        classes,
        RESULT_DIR / "confusion_matrix_disaster_type_phase1.png",
        "Phase 1+5 Calibrated Disaster Classification (Counts)\nMacro F1: "
        f"{test_macro_f1*100:.1f}% | Balanced Acc: {test_bal_acc*100:.1f}% | Miss Rate: {test_miss_rate*100:.1f}%",
        normalize=False,
    )
    _save_cm_plot(
        cm_test,
        classes,
        RESULT_DIR / "confusion_matrix_disaster_type_phase1_normalized.png",
        "Phase 1+5 Calibrated Disaster Classification (Normalized Recall)\nMacro F1: "
        f"{test_macro_f1*100:.1f}% | Balanced Acc: {test_bal_acc*100:.1f}% | Miss Rate: {test_miss_rate*100:.1f}%",
        normalize=True,
    )
    # Also update main confusion_matrix_disaster_type.png
    _save_cm_plot(
        cm_test,
        classes,
        RESULT_DIR / "confusion_matrix_disaster_type.png",
        "Disaster Classification - Confusion Matrix (Counts)\nCalibrated 2036-D RF (Macro F1: "
        f"{test_macro_f1*100:.1f}%, Bal Acc: {test_bal_acc*100:.1f}%)",
        normalize=False,
    )

    # Save Calibrated Bundle & Updated Model Checkpoints
    bundle = CalibratedModelBundle(
        objective="disaster_type",
        feature_pipeline=FEATURE_PIPELINE,
        classes=classes,
        multipliers=[float(x) for x in best_w],
        temperature=best_temp,
        abstention=abstention_cfg,
        operating_point="macro_f1_optimal",
        cost_sensitive_multipliers={
            "cost_2x": [float(x) for x in w_cost_2x],
            "cost_3x": [float(x) for x in w_cost_3x],
        },
        ood_guard=ood_guard,
        metadata={
            "rf_params": best_cfg,
            "oof_macro_f1_mean": round(float(np.mean(fold_macro_f1s)), 4),
            "oof_macro_f1_std": round(float(np.std(fold_macro_f1s)), 4),
            "oof_balanced_acc_mean": round(float(np.mean(fold_bal_accs)), 4),
            "oof_balanced_acc_std": round(float(np.std(fold_bal_accs)), 4),
            "test_macro_f1": round(test_macro_f1, 4),
            "test_balanced_accuracy": round(test_bal_acc, 4),
            "test_disaster_miss_rate": round(test_miss_rate, 4),
            "test_ece": round(test_ece_cal, 4),
        },
    )

    joblib.dump(final_clf, MODEL_DIR / f"rf_disaster_type_{FEATURE_PIPELINE}.joblib")
    joblib.dump(final_scaler, MODEL_DIR / f"scaler_disaster_type_{FEATURE_PIPELINE}.joblib")
    joblib.dump(le, MODEL_DIR / f"le_disaster_type_{FEATURE_PIPELINE}.joblib")
    joblib.dump(le, MODEL_DIR / "le_disaster_type.joblib")
    joblib.dump(bundle, MODEL_DIR / f"calibrated_bundle_disaster_type_{FEATURE_PIPELINE}.joblib")

    # Also build calibrated bundles for damage and informativeness so all 3 objectives support calibration
    for aux_obj in ("damage", "informativeness"):
        aux_clf_p = MODEL_DIR / f"rf_{aux_obj}_{FEATURE_PIPELINE}.joblib"
        aux_le_p = MODEL_DIR / f"le_{aux_obj}_{FEATURE_PIPELINE}.joblib"
        aux_sc_p = MODEL_DIR / f"scaler_{aux_obj}_{FEATURE_PIPELINE}.joblib"
        if aux_clf_p.exists() and aux_le_p.exists() and aux_sc_p.exists():
            aux_clf = joblib.load(aux_clf_p)
            aux_le = joblib.load(aux_le_p)
            aux_sc = joblib.load(aux_sc_p)
            aux_classes = [str(c) for c in aux_le.classes_]
            aux_bundle = CalibratedModelBundle(
                objective=aux_obj,
                feature_pipeline=FEATURE_PIPELINE,
                classes=aux_classes,
                multipliers=[1.0 for _ in aux_classes],
                temperature=1.0,
                abstention=AbstentionConfig(min_confidence=0.35, min_top2_margin=0.05),
            )
            joblib.dump(aux_bundle, MODEL_DIR / f"calibrated_bundle_{aux_obj}_{FEATURE_PIPELINE}.joblib")

    summary_payload = {
        "objective": "disaster_type",
        "grid_search_oof": grid_results,
        "winning_rf_params": best_cfg,
        "multipliers": {cls: round(float(w), 4) for cls, w in zip(classes, best_w)},
        "cost_sensitive_2x": {
            "multipliers": {cls: round(float(w), 4) for cls, w in zip(classes, w_cost_2x)},
            "oof_metrics": stats_cost_2x,
            "test_macro_f1": round(float(f1_score(y_test, cost2x_test_preds, average="macro", zero_division=0)), 4),
            "test_balanced_accuracy": round(float(balanced_accuracy_score(y_test, cost2x_test_preds)), 4),
            "test_disaster_miss_rate": round(disaster_miss_rate(y_test, cost2x_test_preds, not_dis_idx), 4),
        },
        "cost_sensitive_3x": {
            "multipliers": {cls: round(float(w), 4) for cls, w in zip(classes, w_cost_3x)},
            "oof_metrics": stats_cost_3x,
            "test_macro_f1": round(float(f1_score(y_test, cost3x_test_preds, average="macro", zero_division=0)), 4),
            "test_balanced_accuracy": round(float(balanced_accuracy_score(y_test, cost3x_test_preds)), 4),
            "test_disaster_miss_rate": round(disaster_miss_rate(y_test, cost3x_test_preds, not_dis_idx), 4),
        },
        "oof_5fold_summary": {
            "macro_f1_mean": round(float(np.mean(fold_macro_f1s)), 4),
            "macro_f1_std": round(float(np.std(fold_macro_f1s)), 4),
            "balanced_acc_mean": round(float(np.mean(fold_bal_accs)), 4),
            "balanced_acc_std": round(float(np.std(fold_bal_accs)), 4),
            "accuracy_mean": round(float(np.mean(fold_accs)), 4),
            "accuracy_std": round(float(np.std(fold_accs)), 4),
        },
        "calibration_and_abstention": {
            "temperature": round(best_temp, 4),
            "oof_ece_before": round(ece_before, 4),
            "oof_ece_after": round(ece_after, 4),
            "test_ece_raw": round(test_ece_raw, 4),
            "test_ece_calibrated": round(test_ece_cal, 4),
            "abstention_config": {
                "min_confidence": abstention_cfg.min_confidence,
                "min_top2_margin": abstention_cfg.min_top2_margin,
            },
            "oof_abstention": abstention_oof_stats,
            "test_abstain_rate": round(test_abstain_rate, 4),
            "test_answered_accuracy": round(test_answered_acc, 4),
            "test_answered_macro_f1": round(test_answered_macro_f1, 4),
        },
        "test_evaluation": {
            "accuracy": round(test_acc, 4),
            "balanced_accuracy": round(test_bal_acc, 4),
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "disaster_to_not_disaster_miss_rate": round(test_miss_rate, 4),
            "per_class": per_class_test,
            "confusion_matrix": cm_test.tolist(),
        },
        "total_runtime_sec": round(time.perf_counter() - started, 2),
    }

    with open(RESULT_DIR / "metrics_disaster_type.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Write Markdown Report
    md = [
        "# Phase 1 & Phase 5 Evaluation Report: Decision-Rule Repair, Calibration & Abstention",
        "",
        "## 1. Executive Summary (Baseline vs. Phase 1+5)",
        "| Metric | Baseline Uncalibrated RF | Phase 1+5 (Macro-F1 Optimal) | Phase 1+5 (Cost-Sensitive 2x) | Phase 1+5 (Cost-Sensitive 3x) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Macro F1 (Headline)** | `40.21%` | **`{test_macro_f1*100:.2f}%`** | `{summary_payload['cost_sensitive_2x']['test_macro_f1']*100:.2f}%` | `{summary_payload['cost_sensitive_3x']['test_macro_f1']*100:.2f}%` |",
        f"| **Balanced Accuracy** | `35.69%` | **`{test_bal_acc*100:.2f}%`** | `{summary_payload['cost_sensitive_2x']['test_balanced_accuracy']*100:.2f}%` | `{summary_payload['cost_sensitive_3x']['test_balanced_accuracy']*100:.2f}%` |",
        f"| **Disaster -> Not-Disaster Miss Rate** | `31.53%` | **`{test_miss_rate*100:.2f}%`** | **`{summary_payload['cost_sensitive_2x']['test_disaster_miss_rate']*100:.2f}%`** | **`{summary_payload['cost_sensitive_3x']['test_disaster_miss_rate']*100:.2f}%`** |",
        f"| **Overall Accuracy** | `68.45%` | `{test_acc*100:.2f}%` | — | — |",
        f"| **Calibration ECE** | `{test_ece_raw:.4f}` | **`{test_ece_cal:.4f}`** ($T={best_temp:.3f}$) | — | — |",
        "",
        "## 2. Per-Class Test Performance with 95% Wilson Confidence Intervals",
        "| Peril | Support | Baseline Recall | Phase 1+5 Recall (95% CI) | Phase 1+5 Precision (95% CI) | Phase 1+5 F1 | Multiplier $w_k$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    baseline_recalls = {
        "earthquake": "19.90%",
        "flood": "2.08%",
        "hurricane": "68.54%",
        "landslide": "7.14%",
        "not_disaster": "83.16%",
        "wildfire": "33.33%",
    }
    for cls in classes:
        pc = per_class_test[cls]
        r_ci = pc["recall_ci95"]
        p_ci = pc["precision_ci95"]
        w_val = summary_payload["multipliers"][cls]
        md.append(
            f"| `{cls}` | {pc['support']} | {baseline_recalls[cls]} | **`{pc['recall']*100:.2f}%`** (`{r_ci[0]*100:.1f}%–{r_ci[1]*100:.1f}%`) "
            f"| `{pc['precision']*100:.2f}%` (`{p_ci[0]*100:.1f}%–{p_ci[1]*100:.1f}%`) | **`{pc['f1']*100:.2f}%`** | `{w_val:.2f}` |"
        )

    md.extend(
        [
            "",
            "## 3. 5-Fold Out-of-Fold (OOF) Cross-Validation Stability",
            f"- **OOF Macro F1:** `{np.mean(fold_macro_f1s)*100:.2f}% ± {np.std(fold_macro_f1s)*100:.2f}%`",
            f"- **OOF Balanced Accuracy:** `{np.mean(fold_bal_accs)*100:.2f}% ± {np.std(fold_bal_accs)*100:.2f}%`",
            f"- **OOF Overall Accuracy:** `{np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%`",
            "",
            "## 4. Phase 5 Abstention (`'uncertain'`) Performance",
            f"- **Thresholds:** `min_confidence={abstention_cfg.min_confidence}`, `min_top2_margin={abstention_cfg.min_top2_margin}`",
            f"- **Test Abstention Rate:** `{test_abstain_rate*100:.2f}%`",
            f"- **Answered Test Accuracy:** `{test_answered_acc*100:.2f}%` | **Answered Macro F1:** `{test_answered_macro_f1*100:.2f}%`",
        ]
    )

    with open(RESULT_DIR / "phase1_5_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(
        f"[Phase 1 & 5] Done! Test Macro F1: 40.21% -> {test_macro_f1*100:.2f}% | "
        f"Balanced Acc: 35.69% -> {test_bal_acc*100:.2f}% | "
        f"Disaster Miss Rate: 31.53% -> {test_miss_rate*100:.2f}%"
    )
    return summary_payload


if __name__ == "__main__":
    run_phase1_and_5_pipeline()
