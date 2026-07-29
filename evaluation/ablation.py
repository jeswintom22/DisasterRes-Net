"""Automatic ablation and validation experiments for DisasterRes-Net."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from config.research_config import DEFAULT_CONFIG
from features.feature_contract import (
    FeaturePipelineSpec,
    build_feature_matrix,
    get_pipeline_spec,
    validate_feature_importance_length,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE_DIR = ROOT / ".cache" / "feature_cache"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)
ABLATION_TREES = int(os.environ.get("DISASTERRES_ABLATION_TREES", "120"))
PERMUTATION_REPEATS = int(os.environ.get("DISASTERRES_PERMUTATION_REPEATS", "2"))
PERMUTATION_SAMPLE_SIZE = int(os.environ.get("DISASTERRES_PERMUTATION_SAMPLE_SIZE", "250"))
PERMUTATION_MAX_FEATURES = int(os.environ.get("DISASTERRES_PERMUTATION_MAX_FEATURES", "80"))


def _markdown_table(df: pd.DataFrame) -> str:
    """Render markdown tables without optional tabulate dependency."""
    if df.empty:
        return "_No rows._"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class AblationResult:
    """Summary row for one ablation configuration."""

    objective: str
    configuration: str
    rank: int
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    f1_macro: float
    balanced_accuracy: float
    roc_auc: float | None
    feature_dimension: int
    train_images: int
    test_images: int
    training_time_sec: float
    inference_time_ms: float
    memory_mb: float


def _cache_candidates(objective: str, split: str) -> List[Path]:
    return sorted(FEATURE_CACHE_DIR.glob(f"{objective}_{split}_*_features.npz"), key=lambda path: path.stat().st_mtime, reverse=True)


def _load_split_cache(objective: str, split: str) -> Dict[str, np.ndarray]:
    for path in _cache_candidates(objective, split):
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)
        if {"labels", "df1", "df2", "glcm", "lbp"}.issubset(keys):
            return {key: data[key] for key in data.files} | {"__path__": np.asarray(str(path), dtype=object)}
    raise FileNotFoundError(
        f"No cached feature matrix found for objective='{objective}', split='{split}'. "
        "Run step3_classify.py first to generate DF1/DF2/GLCM/LBP caches."
    )


def _available_features(cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    lbp = np.asarray(cache["lbp"], dtype=np.float32)
    features = {
        "cnn_original": np.asarray(cache["df1"], dtype=np.float32),
        "cnn_saliency": np.asarray(cache["df2"], dtype=np.float32),
        "glcm": np.asarray(cache["glcm"], dtype=np.float32),
    }
    if lbp.ndim == 2 and lbp.shape[1] == 10:
        features["lbp_hist"] = lbp
        features["lbp_stats"] = np.hstack([lbp, _lbp_stats_from_hist(lbp)]).astype(np.float32)
    elif lbp.ndim == 2 and lbp.shape[1] == 16:
        features["lbp_stats"] = lbp
        features["lbp_hist"] = lbp[:, :10]
    else:
        raise ValueError(f"Unsupported cached LBP shape: {lbp.shape}")
    return features


def _lbp_stats_from_hist(hist: np.ndarray) -> np.ndarray:
    bins = np.arange(hist.shape[1], dtype=np.float32)
    mean = np.sum(hist * bins, axis=1)
    variance = np.sum(hist * np.square(bins[None, :] - mean[:, None]), axis=1)
    entropy = -np.sum(hist * np.log2(np.clip(hist, 1e-10, 1.0)), axis=1)
    energy = np.sum(np.square(hist), axis=1)
    dominant = np.argmax(hist, axis=1) / max(hist.shape[1] - 1, 1)
    uniformity = np.sum(hist[:, :9], axis=1)
    return np.column_stack([mean, variance, entropy, energy, dominant, uniformity]).astype(np.float32)


def _spec_for_ablation(config_name: str, lbp_width: int) -> FeaturePipelineSpec:
    if lbp_width == 10:
        mapping = {
            "lbp": "lbp_hist",
            "cnn_lbp": "df1_df2_glcm_lbp",  # reduced below by custom spec
            "glcm_lbp": "glcm_lbp",
            "cnn_glcm_lbp": "df1_df2_glcm_lbp",
        }
    else:
        mapping = {}
    if config_name in {"cnn", "glcm", "lbp", "cnn_glcm", "glcm_lbp", "cnn_lbp", "cnn_glcm_lbp"}:
        if config_name == "cnn_glcm_lbp":
            return get_pipeline_spec("df1_df2_glcm_lbp_stats" if lbp_width == 16 else "df1_df2_glcm_lbp")
        if config_name == "lbp" and lbp_width == 10:
            return get_pipeline_spec("lbp_hist")
        return get_pipeline_spec(config_name)
    raise ValueError(f"Unknown ablation configuration: {config_name}")


def _custom_matrix(features: Dict[str, np.ndarray], config_name: str, lbp_width: int) -> Tuple[np.ndarray, List[str]]:
    if config_name == "cnn_lbp" and lbp_width == 10:
        spec = FeaturePipelineSpec("cnn_lbp_legacy", ("cnn_original", "cnn_saliency", "lbp_hist"))
    elif config_name == "glcm_lbp" and lbp_width == 10:
        spec = FeaturePipelineSpec("glcm_lbp_legacy", ("glcm", "lbp_hist"))
    else:
        spec = _spec_for_ablation(config_name, lbp_width)
    return build_feature_matrix(features, spec), spec.feature_names


def run_ablation(objective: str = "informativeness") -> List[AblationResult]:
    """Run all configured feature-fusion ablations for one objective."""
    train_cache = _load_split_cache(objective, "train")
    test_cache = _load_split_cache(objective, "test")
    train_features = _available_features(train_cache)
    test_features = _available_features(test_cache)
    lbp_width = int(np.asarray(train_cache["lbp"]).shape[1])

    le = LabelEncoder()
    y_train = le.fit_transform(train_cache["labels"].astype(str))
    y_test = le.transform(test_cache["labels"].astype(str))
    rows: List[AblationResult] = []
    detail: Dict[str, object] = {"objective": objective, "classes": le.classes_.tolist(), "models": {}}

    for config_name in DEFAULT_CONFIG.ablation_configs:
        x_train_raw, feature_names = _custom_matrix(train_features, config_name, lbp_width)
        x_test_raw, _ = _custom_matrix(test_features, config_name, lbp_width)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)
        clf = RandomForestClassifier(
            n_estimators=ABLATION_TREES,
            max_features="sqrt",
            class_weight="balanced",
            random_state=DEFAULT_CONFIG.random_seed,
            n_jobs=-1,
        )
        start = time.perf_counter()
        clf.fit(x_train, y_train)
        training_time = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = clf.predict(x_test)
        inference_ms = (time.perf_counter() - start) * 1000.0 / max(len(y_test), 1)
        proba = clf.predict_proba(x_test) if hasattr(clf, "predict_proba") else None
        roc_auc = _roc_auc(y_test, proba, len(le.classes_)) if proba is not None else None
        result = AblationResult(
            objective=objective,
            configuration=config_name,
            rank=0,
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision_weighted=float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            recall_weighted=float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            f1_weighted=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
            roc_auc=roc_auc,
            feature_dimension=int(x_train.shape[1]),
            train_images=int(len(y_train)),
            test_images=int(len(y_test)),
            training_time_sec=float(training_time),
            inference_time_ms=float(inference_ms),
            memory_mb=float((x_train_raw.nbytes + x_test_raw.nbytes) / (1024 * 1024)),
        )
        rows.append(result)
        detail["models"][config_name] = {
            "result": asdict(result),
            "classification_report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_names": feature_names,
        }
        _save_feature_importance(objective, config_name, clf, x_test, y_test, feature_names)
        _save_curves(objective, config_name, y_test, proba, le.classes_)

    ranked = sorted(rows, key=lambda row: (row.f1_macro, row.balanced_accuracy, row.accuracy), reverse=True)
    ranked = [AblationResult(**{**asdict(row), "rank": idx + 1}) for idx, row in enumerate(ranked)]
    _save_ablation_outputs(objective, ranked, detail)
    return ranked


def _roc_auc(y_true: np.ndarray, proba: np.ndarray, class_count: int) -> float | None:
    try:
        if class_count == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
    except ValueError:
        return None


def _save_feature_importance(
    objective: str,
    config_name: str,
    clf: RandomForestClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> None:
    validate_feature_importance_length(clf.feature_importances_, FeaturePipelineSpec(config_name, tuple())) if False else None
    if len(feature_names) != len(clf.feature_importances_):
        raise ValueError(
            f"Feature importance mismatch for {config_name}: {len(feature_names)} names vs "
            f"{len(clf.feature_importances_)} importances."
        )
    df = pd.DataFrame({"feature": feature_names, "gini_importance": clf.feature_importances_})
    try:
        if PERMUTATION_REPEATS <= 0:
            raise RuntimeError("Permutation importance disabled by DISASTERRES_PERMUTATION_REPEATS=0")
        if len(y_test) > PERMUTATION_SAMPLE_SIZE:
            rng = np.random.default_rng(DEFAULT_CONFIG.random_seed)
            sample_idx = rng.choice(len(y_test), size=PERMUTATION_SAMPLE_SIZE, replace=False)
            x_perm = x_test[sample_idx]
            y_perm = y_test[sample_idx]
        else:
            x_perm = x_test
            y_perm = y_test
        perm_mean = np.full(len(feature_names), np.nan, dtype=np.float32)
        perm_std = np.full(len(feature_names), np.nan, dtype=np.float32)
        candidate_idx = np.argsort(clf.feature_importances_)[::-1][: min(PERMUTATION_MAX_FEATURES, len(feature_names))]
        baseline = accuracy_score(y_perm, clf.predict(x_perm))
        rng = np.random.default_rng(DEFAULT_CONFIG.random_seed)
        for feature_idx in candidate_idx:
            drops = []
            for _ in range(PERMUTATION_REPEATS):
                shuffled = x_perm.copy()
                shuffled[:, feature_idx] = rng.permutation(shuffled[:, feature_idx])
                drops.append(baseline - accuracy_score(y_perm, clf.predict(shuffled)))
            perm_mean[feature_idx] = float(np.mean(drops))
            perm_std[feature_idx] = float(np.std(drops))
        df["permutation_importance_mean"] = perm_mean
        df["permutation_importance_std"] = perm_std
    except Exception as exc:
        df["permutation_importance_error"] = str(exc)
    df = df.sort_values("gini_importance", ascending=False)
    out_csv = RESULT_DIR / f"feature_importance_{objective}_{config_name}.csv"
    df.to_csv(out_csv, index=False)
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["gini_importance"])
    plt.title(f"Top Feature Importances - {objective} - {config_name}")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"feature_importance_{objective}_{config_name}.png", dpi=180)
    plt.close()

    try:
        import shap  # type: ignore

        sample = x_test[: min(100, len(x_test))]
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(sample)
        np.savez_compressed(RESULT_DIR / f"shap_values_{objective}_{config_name}.npz", values=values, feature_names=np.asarray(feature_names, dtype=object))
    except Exception as exc:
        (RESULT_DIR / f"shap_values_{objective}_{config_name}.md").write_text(
            f"SHAP values were not generated for `{objective}/{config_name}`: {exc}\n",
            encoding="utf-8",
        )


def _save_curves(objective: str, config_name: str, y_test: np.ndarray, proba: np.ndarray | None, classes: np.ndarray) -> None:
    if proba is None:
        return
    try:
        y_bin = label_binarize(y_test, classes=np.arange(len(classes)))
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            precision, recall, _ = precision_recall_curve(y_test, proba[:, 1])
            curves = [(classes[1], fpr, tpr, precision, recall)]
        else:
            curves = []
            for idx, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, idx], proba[:, idx])
                precision, recall, _ = precision_recall_curve(y_bin[:, idx], proba[:, idx])
                curves.append((cls, fpr, tpr, precision, recall))
        plt.figure(figsize=(7, 5))
        for cls, fpr, tpr, _, _ in curves:
            plt.plot(fpr, tpr, label=str(cls))
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.title(f"ROC Curves - {objective} - {config_name}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"roc_curves_{objective}_{config_name}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        for cls, _, _, precision, recall in curves:
            plt.plot(recall, precision, label=str(cls))
        plt.title(f"Precision-Recall - {objective} - {config_name}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"precision_recall_{objective}_{config_name}.png", dpi=180)
        plt.close()
    except Exception as exc:
        (RESULT_DIR / f"curves_{objective}_{config_name}.md").write_text(str(exc), encoding="utf-8")


def _save_ablation_outputs(objective: str, rows: List[AblationResult], detail: Dict[str, object]) -> None:
    df = pd.DataFrame([asdict(row) for row in rows])
    df.to_csv(RESULT_DIR / f"ablation_results_{objective}.csv", index=False)
    df.to_csv(RESULT_DIR / "ablation_results.csv", index=False)
    with open(RESULT_DIR / f"ablation_results_{objective}.json", "w", encoding="utf-8") as fobj:
        json.dump({"ranked_results": [asdict(row) for row in rows], "detail": detail}, fobj, indent=2)
    with open(RESULT_DIR / "ablation_results.json", "w", encoding="utf-8") as fobj:
        json.dump({"ranked_results": [asdict(row) for row in rows], "detail": detail}, fobj, indent=2)
    plt.figure(figsize=(9, 5))
    plt.bar(df["configuration"], df["f1_macro"])
    plt.ylabel("Macro F1")
    plt.title(f"Ablation Ranking - {objective}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"ablation_ranking_{objective}.png", dpi=180)
    plt.close()
    _write_ablation_report(objective, df)


def _write_ablation_report(objective: str, df: pd.DataFrame) -> None:
    lines = [
        f"# Ablation Report - {objective}",
        "",
        "All configurations use the same cached train/test split, RandomForest classifier, scaler policy, and random seed.",
        "",
        _markdown_table(df),
        "",
        "## Ranking Criterion",
        "Models are ranked by Macro F1, then Balanced Accuracy, then Accuracy.",
    ]
    (RESULT_DIR / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULT_DIR / f"ablation_report_{objective}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    objective = os.environ.get("DISASTERRES_OBJECTIVE", "informativeness")
    rows = run_ablation(objective)
    print(pd.DataFrame([asdict(row) for row in rows]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
