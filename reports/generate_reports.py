"""Generate repository audit, architecture, and validation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.feature_contract import LEGACY_PIPELINE_BY_DIM, get_pipeline_spec, resolve_artifact_spec

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "saved_models"
RESULT_DIR.mkdir(exist_ok=True)


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-compatible table without tabulate."""
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


def _scaler_inventory() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(MODEL_DIR.glob("scaler_*.joblib")):
        scaler = joblib.load(path)
        expected = getattr(scaler, "n_features_in_", None)
        pipeline = _pipeline_from_name(path.name)
        try:
            spec = resolve_artifact_spec(pipeline, expected)
            issue = "" if spec.name == pipeline else f"mislabeled: uses {spec.name} contract"
        except Exception as exc:
            spec = None
            issue = str(exc)
        rows.append(
            {
                "artifact": path.name,
                "named_pipeline": pipeline,
                "scaler_features": expected,
                "resolved_pipeline": None if spec is None else spec.name,
                "resolved_dimension": None if spec is None else spec.dimension,
                "issue": issue,
            }
        )
    return rows


def _pipeline_from_name(name: str) -> str:
    if "df1_df2_glcm_lbp_stats" in name:
        return "df1_df2_glcm_lbp_stats"
    if "df1_df2_glcm_lbp" in name:
        return "df1_df2_glcm_lbp"
    return "df1_df2_glcm"


def generate_repository_audit() -> Path:
    inventory = _scaler_inventory()
    df = pd.DataFrame(inventory)
    issues = [row for row in inventory if row["issue"]]
    lines = [
        "# DisasterRes-Net Repository Audit",
        "",
        "## Scope",
        "Traced training, feature extraction, fusion, scaling, checkpoint loading, inference, Flask serialization, visualization, and M2 localization.",
        "",
        "## Critical Findings Before Fixes",
        "",
        "1. Feature contract was global-name based instead of artifact based. A checkpoint named `df1_df2_glcm_lbp_stats` could still contain a 2030-feature scaler.",
        "2. The 2036 vs 2030 runtime error was caused by LBP schema drift: the code emitted 16 LBP features while the scaler expected a legacy 10-bin LBP vector.",
        "3. The LBP cache key did not include feature schema/version, allowing stale 10-feature caches to be reused by stats-based training.",
        "4. Training, demo inference, hybrid Flask inference, and Step 4 localization contained duplicated feature and saliency logic.",
        "5. Feature names were manually assembled and could diverge from `feature_importances_` length.",
        "6. M2 localization used generic saliency thresholds and did not expose backend choice, polygons, region confidence, or paper-reproduction metadata.",
        "7. Dashboard error handling returned only coarse errors and did not expose model/feature/localization metadata needed for research review.",
        "",
        "## Checkpoint Feature Inventory",
        "",
        _markdown_table(df),
        "",
        "## Dimension Diagnosis",
        "",
        "Current registered feature contracts:",
        "",
    ]
    for dim, name in sorted(LEGACY_PIPELINE_BY_DIM.items()):
        spec = get_pipeline_spec(name)
        lines.append(f"- `{name}`: {spec.dimension} features, streams={list(spec.streams)}")
    lines.extend(
        [
            "",
            "The previously observed `2036 features, expected 2030` error is resolved by loading the scaler first, resolving the true feature contract from `n_features_in_`, and extracting either LBP histogram or LBP histogram+statistics accordingly.",
            "",
            "## Fixes Applied",
            "",
            "- Added `features/feature_contract.py` as the single source of truth for stream dimensions, feature names, and artifact compatibility.",
            "- Updated LBP cache hashing and stale-cache validation in `step3_classify.py`.",
            "- Updated `_load_rf_bundle` to return a resolved `FeaturePipelineSpec` based on scaler dimension.",
            "- Updated hybrid inference to build fused matrices through `build_feature_matrix`/`fuse_feature_streams`.",
            "- Replaced manual feature-importance naming with generated feature names and length validation.",
            "- Added configurable localization backends in `damage_assessment/localization_backends.py`.",
            "- Redesigned DDM/DEM in `damage_assessment/localization.py` with adaptive thresholding, morphology, connected components, region confidence, polygons, and richer DEM terms.",
            "- Added cached-feature ablation framework in `evaluation/ablation.py`.",
            "",
            "## Remaining Research Caveats",
            "",
            "- Exact SUN implementation details from the original paper are not fully specified; the repo documents its LMS/intensity/gradient + FastICA approximation.",
            "- GradCAM-family backends require PyTorch/timm model loading and are slower than SUN+ICA/OpenCV.",
            "- Pixel-level localization metrics require ground-truth masks. Without masks, the framework reports qualitative/stability metrics instead.",
        ]
    )
    out = ROOT / "repository_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_architecture_doc() -> Path:
    lines = [
        "# DisasterRes-Net Research Architecture",
        "",
        "## Shared Feature Pipeline",
        "All training, evaluation, inference, and Flask prediction paths use feature contracts from `features/feature_contract.py`.",
        "",
        "Supported configurations:",
        "",
    ]
    for name in ["cnn", "glcm", "lbp", "cnn_lbp", "cnn_glcm", "glcm_lbp", "cnn_glcm_lbp", "df1_df2_glcm", "df1_df2_glcm_lbp", "df1_df2_glcm_lbp_stats"]:
        spec = get_pipeline_spec(name)
        lines.append(f"- `{name}`: {spec.dimension} features, streams={list(spec.streams)}")
    lines.extend(
        [
            "",
            "## M1 Prediction",
            "Image inputs are transformed into CNN original/saliency streams, GLCM descriptors, and LBP descriptors. The selected feature spec determines the fused vector and generated feature names. The scaler dimension is validated before classifier inference.",
            "",
            "## M2 Localization",
            "Backends: `sun_ica`, `opencv`, `gradcam`, `gradcam_plus_plus`, `scorecam`. Each backend produces a normalized activation map consumed by the same DDM/DEM analyzer.",
            "",
            "## DEM Equation",
            "DEM = 100 * severity_weight * (0.22A + 0.18D + 0.13L + 0.09R + 0.10C + 0.08K + 0.07B + 0.06T + 0.04Q + 0.03M), clipped to [0, 100].",
            "",
            "Where A=damage area ratio, D=damage density, L=largest-region ratio, R=average-region-size term, C=region-count term, K=1-compactness, B=boundary complexity, T=texture entropy, Q=localization confidence, M=average activation.",
        ]
    )
    out = ROOT / "architecture.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_validation_report() -> Path:
    ablation_path = RESULT_DIR / "ablation_results.json"
    lines = ["# Research Validation", ""]
    if ablation_path.exists():
        data = json.loads(ablation_path.read_text(encoding="utf-8"))
        rows = data.get("ranked_results", [])
        if rows:
            df = pd.DataFrame(rows)
            lines.extend([
                "## Ablation Ranking",
                "",
                _markdown_table(df),
                "",
                f"Best configuration: `{rows[0]['configuration']}` with Macro F1={rows[0]['f1_macro']:.4f}.",
            ])
    else:
        lines.extend([
            "## Ablation Ranking",
            "",
            "Ablation has not been run in this workspace yet. Execute `python -m evaluation.ablation` after feature caches are available.",
        ])
    lines.extend(
        [
            "",
            "## Error and Failure Analysis",
            "Misclassified, best, and worst predictions are recorded inside the detailed ablation JSON when ablation is run with cached features.",
            "",
            "## Statistical Comparison",
            "The ablation framework uses identical splits and reports macro F1, weighted F1, balanced accuracy, ROC AUC, timing, and memory. Add repeated seeds or bootstrap resampling for formal significance tests in a paper submission.",
        ]
    )
    out = ROOT / "research_validation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_publication_outputs() -> None:
    """Create stable publication filenames from the latest evaluation artifacts."""
    ablation_path = RESULT_DIR / "ablation_results.json"
    if ablation_path.exists():
        data = json.loads(ablation_path.read_text(encoding="utf-8"))
        rows = data.get("ranked_results", [])
        detail = data.get("detail", {}).get("models", {})
        if rows:
            best = rows[0]["configuration"]
            best_detail = detail.get(best, {})
            report = best_detail.get("classification_report")
            if report:
                pd.DataFrame(report).T.to_csv(RESULT_DIR / "classification_report.csv")
            cm = best_detail.get("confusion_matrix")
            classes = data.get("detail", {}).get("classes", [])
            if cm:
                matrix = np.asarray(cm)
                plt.figure(figsize=(7, 6))
                plt.imshow(matrix, cmap="Blues")
                plt.title(f"Confusion Matrix - Best Ablation ({best})")
                plt.colorbar()
                ticks = np.arange(len(classes))
                plt.xticks(ticks, classes, rotation=35, ha="right")
                plt.yticks(ticks, classes)
                for row in range(matrix.shape[0]):
                    for col in range(matrix.shape[1]):
                        plt.text(col, row, int(matrix[row, col]), ha="center", va="center")
                plt.ylabel("True")
                plt.xlabel("Predicted")
                plt.tight_layout()
                plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=180)
                plt.close()
            src = RESULT_DIR / f"feature_importance_{data.get('detail', {}).get('objective', 'informativeness')}_{best}.csv"
            if src.exists():
                pd.read_csv(src).to_csv(RESULT_DIR / "feature_importance.csv", index=False)
            src_png = RESULT_DIR / f"feature_importance_{data.get('detail', {}).get('objective', 'informativeness')}_{best}.png"
            if src_png.exists():
                (RESULT_DIR / "feature_importance.png").write_bytes(src_png.read_bytes())

    m2_path = RESULT_DIR / "m2_report_test.json"
    if m2_path.exists():
        try:
            data = json.loads(m2_path.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            pd.DataFrame([metrics]).to_csv(RESULT_DIR / "dem_statistics.csv", index=False)
        except Exception as exc:
            (RESULT_DIR / "dem_statistics_error.md").write_text(str(exc), encoding="utf-8")


def generate_final_project_report() -> Path:
    generate_publication_outputs()
    audit = generate_repository_audit()
    arch = generate_architecture_doc()
    validation = generate_validation_report()
    lines = [
        "# Final Project Report",
        "",
        "## Issues Found",
        "See `repository_audit.md` for the complete audit. Main issues were feature-schema drift, stale LBP caches, duplicated fusion logic, and insufficiently modular localization.",
        "",
        "## Fixes Applied",
        "- Shared feature contracts and automatic dimension validation.",
        "- Legacy checkpoint compatibility based on scaler dimensions.",
        "- LBP cache versioning and stale-cache rejection.",
        "- Configurable feature fusion and ablation configurations.",
        "- Modular localization backends including SUN+ICA and CAM family methods.",
        "- Redesigned DDM/DEM with documented equation and richer region metrics.",
        "- Publication-output generators for audit, architecture, ablation, feature importance, and validation artifacts.",
        "",
        "## Architectural Improvements",
        "The framework now separates preprocessing, feature contracts, fusion, model orchestration, localization, evaluation, visualization, reports, and utilities.",
        "",
        "## Research Improvements",
        "LBP remains a first-class research contribution and is evaluated alone and in fused configurations. Localization can compare paper baseline, current saliency, and modern CAM methods.",
        "",
        "## Validation Results",
        f"Generated: `{audit.name}`, `{arch.name}`, `{validation.name}`. Run `python -m evaluation.ablation` to refresh quantitative tables from cached features.",
        "",
        "## Remaining Limitations",
        "Exact paper SUN parameters are under-specified; GradCAM-family methods require heavier model loading; pixel-level quality metrics require mask annotations.",
        "",
        "## Future Work",
        "Add annotated localization masks, bootstrap significance testing, calibration curves, external disaster datasets, and end-to-end neural fusion training.",
    ]
    out = ROOT / "FINAL_PROJECT_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    for path in [generate_repository_audit(), generate_architecture_doc(), generate_validation_report(), generate_final_project_report()]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
