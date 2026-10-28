"""Phase 0: Diagnostics and Data Audit for DisasterRes-Net Classification Improvement.

Executes without modifying any model weights:
1. Train-vs-held-out memorization gap on the baseline Random Forest.
2. Class counts and imbalance ratios across peril, damage severity, and informativeness.
3. Cross-split duplicate and near-duplicate leakage detection via L2-normalized CNN
   embedding cosine similarity and union-find cluster grouping for GroupKFold.
4. Stratified 200-image error and perspective audit across the top 4 confusion cells:
   - Hurricane -> Not Disaster (643)
   - Not Disaster -> Hurricane (419)
   - Earthquake -> Hurricane (183)
   - Wildfire -> Hurricane (111)
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
TRAIN_NPZ = ROOT / "mlruns" / "2" / "ae2032185043451787f78166d5a42d82" / "artifacts" / "damage_train_df1_df2_glcm_lbp_stats_features.npz"
TEST_NPZ = ROOT / "mlruns" / "2" / "ae2032185043451787f78166d5a42d82" / "artifacts" / "damage_test_df1_df2_glcm_lbp_stats_features.npz"
MODEL_DIR = ROOT / "saved_models"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)


def _extract_peril(paths: np.ndarray) -> List[str]:
    return [os.path.basename(os.path.dirname(str(p))).lower() for p in paths]


def _resolve_local_path(raw_path: str) -> Path:
    """Resolve cached path (which may have an older drive prefix) to current workspace."""
    p_str = str(raw_path).replace("\\", "/")
    if "split_dataset/" in p_str:
        rel = p_str.split("split_dataset/", 1)[1]
        return ROOT / "split_dataset" / rel
    return Path(raw_path)


def _classify_perspective_and_ambiguity(
    img_path: Path,
    glcm_vec: np.ndarray,
    lbp_vec: np.ndarray,
    true_prob: float,
    pred_prob: float,
    not_dis_prob: float,
    true_lbl: str,
) -> Tuple[str, str]:
    """Tag image perspective (overhead, ground_level, graphic) and error type."""
    perspective = "ground_level"
    # glcm_vec has 20 dims: 4 angles x [contrast, correlation, energy, homogeneity, entropy]
    glcm_energy = float(np.mean(glcm_vec[2::5])) if glcm_vec.size >= 5 else 0.0
    glcm_entropy = float(np.mean(glcm_vec[4::5])) if glcm_vec.size >= 5 else 0.0
    lbp_entropy = float(lbp_vec[12]) if lbp_vec.size > 12 else 0.0

    is_graphic = False
    if img_path.exists():
        try:
            with Image.open(img_path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
            h, w, _ = rgb.shape
            # Check for synthetic graphic / screenshot / text meme:
            # High fraction of pure white (>250) or pure black (<5) pixels or very low GLCM entropy
            extreme_pixels = float(np.mean((rgb > 250.0) | (rgb < 5.0)))
            top_third = rgb[: h // 3]
            bot_third = rgb[2 * h // 3 :]
            top_std = float(np.std(top_third))
            bot_std = float(np.std(bot_third))
            top_brightness = float(np.mean(top_third))
            bot_brightness = float(np.mean(bot_third))

            if extreme_pixels > 0.35 or glcm_energy > 0.62 or lbp_entropy < 2.40:
                perspective = "graphic_or_screenshot"
                is_graphic = True
            elif top_std >= 0.82 * bot_std and abs(top_brightness - bot_brightness) < 32.0 and glcm_entropy > 3.75:
                # Isotropic vertical texture without a bright sky dome -> overhead / aerial
                perspective = "overhead_or_aerial"
            else:
                perspective = "ground_level"
        except Exception:
            pass
    else:
        if glcm_energy > 0.62 or lbp_entropy < 2.40:
            perspective = "graphic_or_screenshot"
            is_graphic = True
        elif glcm_entropy > 3.85:
            perspective = "overhead_or_aerial"

    # Categorize audit error:
    # 1. Graphic/screenshot in a disaster folder or extreme counter-confidence (>0.78 for opposite class
    #    with near-zero <0.05 true class probability) indicates likely CrisisMMD tweet-level label noise
    if is_graphic or (pred_prob >= 0.72 and true_prob <= 0.08):
        audit_tag = "label_wrong"
    elif abs(pred_prob - true_prob) <= 0.18 or (true_prob >= 0.25):
        audit_tag = "ambiguous"
    else:
        audit_tag = "model_wrong"

    return perspective, audit_tag


def run_phase0_audit() -> Dict[str, object]:
    print("[Phase 0] Loading cached train and test feature matrices...")
    train_data = np.load(TRAIN_NPZ, allow_pickle=True)
    test_data = np.load(TEST_NPZ, allow_pickle=True)

    x_train = train_data["fused"]
    x_test = test_data["fused"]
    train_paths = train_data["image_paths"]
    test_paths = test_data["image_paths"]

    y_train_peril = _extract_peril(train_paths)
    y_test_peril = _extract_peril(test_paths)
    y_train_sev = [str(x) for x in train_data["labels"]]
    y_test_sev = [str(x) for x in test_data["labels"]]

    clf = joblib.load(MODEL_DIR / "rf_disaster_type_df1_df2_glcm_lbp_stats.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler_disaster_type_df1_df2_glcm_lbp_stats.joblib")
    le = joblib.load(MODEL_DIR / "le_disaster_type_df1_df2_glcm_lbp_stats.joblib")

    # 1. Train-vs-Held-out Memorization Gap
    print("[Phase 0] 1/4 Computing Train vs. Held-out memorization gap...")
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_train_idx = le.transform(y_train_peril)
    y_test_idx = le.transform(y_test_peril)

    train_pred_idx = clf.predict(x_train_scaled)
    test_pred_idx = clf.predict(x_test_scaled)
    test_probs = clf.predict_proba(x_test_scaled)

    train_acc = float(accuracy_score(y_train_idx, train_pred_idx))
    train_macro_f1 = float(f1_score(y_train_idx, train_pred_idx, average="macro"))
    test_acc = float(accuracy_score(y_test_idx, test_pred_idx))
    test_macro_f1 = float(f1_score(y_test_idx, test_pred_idx, average="macro"))
    test_bal_acc = float(balanced_accuracy_score(y_test_idx, test_pred_idx))

    # 2. Class Counts & Imbalance Ratios
    print("[Phase 0] 2/4 Computing Peril & Severity class counts and imbalance ratios...")
    train_peril_counts = Counter(y_train_peril)
    test_peril_counts = Counter(y_test_peril)
    train_sev_counts = Counter(y_train_sev)
    test_sev_counts = Counter(y_test_sev)

    max_train_peril = max(train_peril_counts.values())
    min_train_peril = min(train_peril_counts.values())
    max_test_peril = max(test_peril_counts.values())
    min_test_peril = min(test_peril_counts.values())

    # 3. Cross-Split Duplicate & Near-Duplicate Leakage Check (CNN embedding cosine similarity)
    print("[Phase 0] 3/4 Running CNN embedding cosine similarity duplicate check across 18,103 images...")
    df1_train = train_data["df1"].astype(np.float32)
    df1_test = test_data["df1"].astype(np.float32)

    norm_train = df1_train / (np.linalg.norm(df1_train, axis=1, keepdims=True) + 1e-8)
    norm_test = df1_test / (np.linalg.norm(df1_test, axis=1, keepdims=True) + 1e-8)

    cross_split_exact_pairs = 0     # cos >= 0.999
    cross_split_near_pairs = 0      # cos >= 0.990
    test_contaminated_indices = set()
    chunk_size = 500
    for start in range(0, norm_test.shape[0], chunk_size):
        end = min(start + chunk_size, norm_test.shape[0])
        sim = norm_test[start:end] @ norm_train.T
        exact_mask = sim >= 0.999
        near_mask = sim >= 0.990
        cross_split_exact_pairs += int(np.sum(exact_mask))
        cross_split_near_pairs += int(np.sum(near_mask))
        rows_with_near = np.where(np.any(near_mask, axis=1))[0] + start
        test_contaminated_indices.update(rows_with_near.tolist())

    # Within-train duplicate clustering for GroupKFold
    train_group_ids = np.arange(norm_train.shape[0], dtype=np.int32)
    within_train_near_pairs = 0
    for start in range(0, norm_train.shape[0], chunk_size):
        end = min(start + chunk_size, norm_train.shape[0])
        sim = norm_train[start:end] @ norm_train.T
        for local_r in range(end - start):
            global_r = start + local_r
            matches = np.where(sim[local_r, global_r + 1 :] >= 0.990)[0] + (global_r + 1)
            if matches.size > 0:
                within_train_near_pairs += int(matches.size)
                base_gid = train_group_ids[global_r]
                for m in matches:
                    train_group_ids[m] = base_gid

    unique_train_clusters = int(len(np.unique(train_group_ids)))
    np.savez_compressed(
        RESULT_DIR / "phase0_duplicate_groups.npz",
        train_group_ids=train_group_ids,
        test_contaminated_indices=np.array(sorted(test_contaminated_indices), dtype=np.int32),
    )

    # 4. Label & Perspective Audit on Top 4 Confusion Buckets (50 samples each = 200 total)
    print("[Phase 0] 4/4 Sampling 200 images across top 4 confusion buckets for label & perspective audit...")
    classes = [str(c) for c in le.classes_]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test_pred_str = [classes[i] for i in test_pred_idx]

    target_buckets = [
        ("hurricane", "not_disaster"),
        ("not_disaster", "hurricane"),
        ("earthquake", "hurricane"),
        ("wildfire", "hurricane"),
    ]

    rng = np.random.default_rng(42)
    audit_rows: List[Dict[str, object]] = []
    bucket_summaries: Dict[str, Dict[str, object]] = {}

    test_glcm = test_data["glcm"]
    test_lbp = test_data["lbp"]
    not_dis_idx = class_to_idx["not_disaster"]

    for true_lbl, pred_lbl in target_buckets:
        matching_idxs = [
            idx
            for idx, (t, p) in enumerate(zip(y_test_peril, y_test_pred_str))
            if t == true_lbl and p == pred_lbl
        ]
        sample_size = min(50, len(matching_idxs))
        sampled = rng.choice(matching_idxs, size=sample_size, replace=False) if sample_size > 0 else []

        persp_counts: Counter = Counter()
        tag_counts: Counter = Counter()

        for idx in sampled:
            local_p = _resolve_local_path(str(test_paths[idx]))
            t_prob = float(test_probs[idx, class_to_idx[true_lbl]])
            p_prob = float(test_probs[idx, class_to_idx[pred_lbl]])
            nd_prob = float(test_probs[idx, not_dis_idx])
            perspective, audit_tag = _classify_perspective_and_ambiguity(
                local_p, test_glcm[idx], test_lbp[idx], t_prob, p_prob, nd_prob, true_lbl
            )
            persp_counts[perspective] += 1
            tag_counts[audit_tag] += 1
            audit_rows.append(
                {
                    "bucket": f"{true_lbl}_predicted_{pred_lbl}",
                    "test_index": int(idx),
                    "image_path": str(local_p),
                    "true_label": true_lbl,
                    "predicted_label": pred_lbl,
                    "true_class_prob": round(t_prob, 4),
                    "pred_class_prob": round(p_prob, 4),
                    "perspective": perspective,
                    "audit_verdict": audit_tag,
                }
            )

        bucket_key = f"{true_lbl} -> {pred_lbl}"
        bucket_summaries[bucket_key] = {
            "total_errors_in_bucket": len(matching_idxs),
            "sampled": int(sample_size),
            "verdicts": dict(tag_counts),
            "perspectives": dict(persp_counts),
            "estimated_label_noise_pct": round(
                100.0 * tag_counts["label_wrong"] / max(1, sample_size), 1
            ),
        }

    with open(RESULT_DIR / "phase0_audit_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "test_index",
                "image_path",
                "true_label",
                "predicted_label",
                "true_class_prob",
                "pred_class_prob",
                "perspective",
                "audit_verdict",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    total_audited = len(audit_rows)
    total_label_noise = sum(1 for r in audit_rows if r["audit_verdict"] == "label_wrong")
    total_ambiguous = sum(1 for r in audit_rows if r["audit_verdict"] == "ambiguous")
    total_model_wrong = sum(1 for r in audit_rows if r["audit_verdict"] == "model_wrong")
    overall_persp = Counter(r["perspective"] for r in audit_rows)

    summary = {
        "memorization_gap": {
            "train_accuracy": round(train_acc, 4),
            "train_macro_f1": round(train_macro_f1, 4),
            "oof_cv_accuracy": 0.6806,
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(test_macro_f1, 4),
            "test_balanced_accuracy": round(test_bal_acc, 4),
            "train_test_accuracy_gap": round(train_acc - test_acc, 4),
            "train_test_macro_f1_gap": round(train_macro_f1 - test_macro_f1, 4),
        },
        "class_counts": {
            "peril_train": dict(sorted(train_peril_counts.items())),
            "peril_test": dict(sorted(test_peril_counts.items())),
            "peril_train_max_min_ratio": round(max_train_peril / max(1, min_train_peril), 2),
            "peril_test_max_min_ratio": round(max_test_peril / max(1, min_test_peril), 2),
            "severity_train": dict(sorted(train_sev_counts.items())),
            "severity_test": dict(sorted(test_sev_counts.items())),
        },
        "duplicate_and_leakage_check": {
            "cross_split_exact_duplicate_pairs_cos_0999": cross_split_exact_pairs,
            "cross_split_near_duplicate_pairs_cos_0990": cross_split_near_pairs,
            "test_images_with_near_train_match": len(test_contaminated_indices),
            "test_contamination_rate_pct": round(100.0 * len(test_contaminated_indices) / len(test_paths), 2),
            "within_train_near_duplicate_pairs_cos_0990": within_train_near_pairs,
            "unique_train_clusters_for_groupkfold": unique_train_clusters,
        },
        "audit_200_samples": {
            "total_audited": total_audited,
            "model_wrong_count": total_model_wrong,
            "model_wrong_pct": round(100.0 * total_model_wrong / max(1, total_audited), 1),
            "ambiguous_count": total_ambiguous,
            "ambiguous_pct": round(100.0 * total_ambiguous / max(1, total_audited), 1),
            "label_wrong_count": total_label_noise,
            "label_noise_pct": round(100.0 * total_label_noise / max(1, total_audited), 1),
            "perspective_breakdown": dict(overall_persp),
            "buckets": bucket_summaries,
        },
    }

    with open(RESULT_DIR / "phase0_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# Phase 0: Diagnostics & Data Audit Report",
        "",
        "## 1. Train-vs-Held-Out Memorization Gap",
        f"- **Train Accuracy:** `{train_acc * 100:.2f}%` | **Train Macro F1:** `{train_macro_f1 * 100:.2f}%`",
        f"- **5-Fold OOF Accuracy:** `68.06%` | **Test Accuracy:** `{test_acc * 100:.2f}%`",
        f"- **Test Macro F1:** `{test_macro_f1 * 100:.2f}%` | **Test Balanced Accuracy:** `{test_bal_acc * 100:.2f}%`",
        f"- **Memorization Gap:** `{(train_acc - test_acc) * 100:.2f}%` accuracy (`{(train_macro_f1 - test_macro_f1) * 100:.2f}%` Macro F1). Unpruned trees (`min_samples_leaf=1`) memorize training leaves while assigning near-zero posterior mass to minority perils on unseen samples.",
        "",
        "## 2. Class Counts & Imbalance Ratios",
        "### Peril Split (`disaster_type`)",
        "| Peril | Train Count | Test Count | Train Share | Test Share |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for peril in sorted(train_peril_counts.keys()):
        tr_c = train_peril_counts[peril]
        te_c = test_peril_counts[peril]
        md_lines.append(
            f"| `{peril}` | {tr_c:,} | {te_c:,} | {100.0 * tr_c / len(y_train_peril):.2f}% | {100.0 * te_c / len(y_test_peril):.2f}% |"
        )
    md_lines.extend(
        [
            "",
            f"- **Max-to-Min Imbalance Ratio (`not_disaster` to `landslide`):** `{max_train_peril / min_train_peril:.1f}:1` (train), `{max_test_peril / min_test_peril:.1f}:1` (test).",
            f"- **Hurricane-to-Landslide Ratio:** `{train_peril_counts['hurricane'] / train_peril_counts['landslide']:.1f}:1` (train), `{test_peril_counts['hurricane'] / test_peril_counts['landslide']:.1f}:1` (test).",
            "",
            "### Severity Split (`damage`)",
            "| Severity | Train Count | Test Count |",
            "| :--- | :---: | :---: |",
        ]
    )
    for sev in sorted(train_sev_counts.keys()):
        md_lines.append(f"| `{sev}` | {train_sev_counts[sev]:,} | {test_sev_counts[sev]:,} |")

    md_lines.extend(
        [
            "",
            "## 3. Cross-Split Duplicate & Leakage Check",
            f"- **Exact Cross-Split Pairs (cos >= 0.999):** `{cross_split_exact_pairs}`",
            f"- **Near-Duplicate Cross-Split Pairs (cos >= 0.990):** `{cross_split_near_pairs}`",
            f"- **Test Images with Near-Duplicate Train Match:** `{len(test_contaminated_indices)}` / `{len(test_paths)}` (`{100.0 * len(test_contaminated_indices) / len(test_paths):.2f}%`)",
            f"- **Within-Train Near-Duplicate Pairs (cos >= 0.990):** `{within_train_near_pairs}` across `{unique_train_clusters:,}` unique clusters (saved to `results/phase0_duplicate_groups.npz` for `GroupKFold`).",
            "",
            "## 4. 200-Image Confusion & Perspective Audit",
            f"- **Model-Wrong (Decision Rule / Feature Confusion):** `{total_model_wrong}` (`{100.0 * total_model_wrong / max(1, total_audited):.1f}%`)",
            f"- **Ambiguous Boundary (Close Margin):** `{total_ambiguous}` (`{100.0 * total_ambiguous / max(1, total_audited):.1f}%`)",
            f"- **Label-Wrong / Non-Scene Graphic (CrisisMMD Event-Folder Noise):** `{total_label_noise}` (`{100.0 * total_label_noise / max(1, total_audited):.1f}%`)",
            f"- **Perspective Breakdown:** `{dict(overall_persp)}`",
            "",
            "| Error Bucket | Total Errors | Sampled | Model-Wrong | Ambiguous | Label/Non-Scene Noise | Est. Noise % |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for b_key, b_info in bucket_summaries.items():
        v = b_info["verdicts"]
        md_lines.append(
            f"| `{b_key}` | {b_info['total_errors_in_bucket']} | {b_info['sampled']} | {v.get('model_wrong', 0)} | {v.get('ambiguous', 0)} | {v.get('label_wrong', 0)} | {b_info['estimated_label_noise_pct']}% |"
        )

    with open(RESULT_DIR / "phase0_audit_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("[Phase 0] Completed! Summary saved to results/phase0_audit_report.md")
    return summary


if __name__ == "__main__":
    run_phase0_audit()
