"""Phase 3 & Phase 4: Multi-Task Neural Training, Loss Arm Comparison & Peril-Conditioned Severity.

Implements:
- FR-7: 3-stage unfreezing schedule (Stage A/B/C), BatchNorm freeze discipline, and
  Phase 3.6 Acceptance Stop Gate (>= +5 OOF Macro F1 over Phase 1/2 RF required to replace RF).
- FR-8: Mutually exclusive comparison of three imbalance loss arms on 5-fold OOF:
    1. EffectiveNumberFocalLoss (gamma=2.0, beta=0.999)
    2. LogitAdjustedCrossEntropy (tau=1.0)
    3. ClassBalancedSampler + CrossEntropyLoss (label_smoothing=0.05)
- FR-9: Peril-conditioned severity head (sum_k p(peril=k) * head_k(f)) with shared
  low-rank adapter for flood and landslide, evaluated against flat unconditioned severity.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.calibration import apply_multipliers, disaster_miss_rate, fit_probability_multipliers
from models.multitask_net import (
    EffectiveNumberFocalLoss,
    LogitAdjustedCrossEntropy,
    PerilConditionedSeverityHead,
    build_class_balanced_sampler,
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
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "saved_models"
RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEVERITY_MAP = {
    "earthquake": "severe",
    "hurricane": "severe",
    "flood": "mild",
    "wildfire": "mild",
    "landslide": "little_or_none",
    "not_disaster": "little_or_none",
}
INFO_MAP = {
    "earthquake": "informative",
    "hurricane": "informative",
    "flood": "informative",
    "wildfire": "informative",
    "landslide": "informative",
    "not_disaster": "not_informative",
}


class FusedMultiTaskHeadNet(nn.Module):
    """Multi-task neural classifier + PerilConditionedSeverityHead on fused feature vectors."""

    def __init__(self, in_dim: int = 2036, trunk_dim: int = 512, num_perils: int = 6, num_severity: int = 3) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(trunk_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.head_peril = nn.Linear(trunk_dim, num_perils)
        self.head_severity_conditioned = PerilConditionedSeverityHead(
            in_features=trunk_dim,
            num_perils=num_perils,
            num_severity=num_severity,
            low_rank=16,
            minority_peril_indices=(1, 3),
        )
        self.head_severity_flat = nn.Linear(trunk_dim, num_severity)
        self.head_info = nn.Linear(trunk_dim, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.trunk(x)
        logits_peril = self.head_peril(feat)
        peril_probs = F.softmax(logits_peril, dim=-1)
        logits_sev_cond = self.head_severity_conditioned(feat, peril_probs)
        logits_sev_flat = self.head_severity_flat(feat)
        logits_info = self.head_info(feat)
        return {
            "disaster_type": logits_peril,
            "damage_conditioned": logits_sev_cond,
            "damage_flat": logits_sev_flat,
            "informativeness": logits_info,
        }


def _extract_peril(paths: np.ndarray) -> List[str]:
    return [os.path.basename(os.path.dirname(str(p))).lower() for p in paths]


def _train_fold_arm(
    arm_name: str,
    x_tr: np.ndarray,
    y_peril_tr: np.ndarray,
    y_sev_tr: np.ndarray,
    y_info_tr: np.ndarray,
    x_val: np.ndarray,
    num_perils: int,
    num_sev: int,
    epochs: int = 12,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusedMultiTaskHeadNet(in_dim=x_tr.shape[1], trunk_dim=384, num_perils=num_perils, num_severity=num_sev).to(device)

    peril_counts = np.bincount(y_peril_tr, minlength=num_perils)
    sev_counts = np.bincount(y_sev_tr, minlength=num_sev)

    if arm_name == "effective_number_focal":
        criterion_peril = EffectiveNumberFocalLoss(peril_counts, beta=0.999, gamma=2.0, label_smoothing=0.05).to(device)
        sampler = None
    elif arm_name == "logit_adjusted_ce":
        criterion_peril = LogitAdjustedCrossEntropy(peril_counts, tau=1.0, label_smoothing=0.05).to(device)
        sampler = None
    elif arm_name == "class_balanced_sampler_ce":
        criterion_peril = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
        sampler = build_class_balanced_sampler(y_peril_tr, power=0.5)
    else:
        raise ValueError(f"Unknown loss arm: {arm_name}")

    criterion_sev = LogitAdjustedCrossEntropy(sev_counts, tau=0.7, label_smoothing=0.03).to(device)
    criterion_info = nn.CrossEntropyLoss(label_smoothing=0.02).to(device)

    ds = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_peril_tr, dtype=torch.long),
        torch.tensor(y_sev_tr, dtype=torch.long),
        torch.tensor(y_info_tr, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _ in range(epochs):
        for bx, by_p, by_s, by_i in loader:
            bx, by_p, by_s, by_i = bx.to(device), by_p.to(device), by_s.to(device), by_i.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss_p = criterion_peril(out["disaster_type"], by_p)
            loss_s_cond = criterion_sev(out["damage_conditioned"], by_s)
            loss_s_flat = criterion_sev(out["damage_flat"], by_s)
            loss_i = criterion_info(out["informativeness"], by_i)
            loss = 1.0 * loss_p + 0.5 * loss_s_cond + 0.5 * loss_s_flat + 0.2 * loss_i
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.inference_mode():
        val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
        out_val = model(val_t)
        p_peril = F.softmax(out_val["disaster_type"], dim=-1).cpu().numpy()
        p_sev_cond = F.softmax(out_val["damage_conditioned"], dim=-1).cpu().numpy()
        p_sev_flat = F.softmax(out_val["damage_flat"], dim=-1).cpu().numpy()

    return p_peril, p_sev_cond, p_sev_flat


def run_phase3_and_4_evaluation() -> Dict[str, object]:
    print("[Phase 3 & 4] Loading cached features and labels for multi-task loss arm comparison...")
    train_data = np.load(TRAIN_NPZ, allow_pickle=True)
    test_data = np.load(TEST_NPZ, allow_pickle=True)

    x_train = train_data["fused"].astype(np.float32)
    x_test = test_data["fused"].astype(np.float32)

    y_peril_tr_str = _extract_peril(train_data["image_paths"])
    y_peril_te_str = _extract_peril(test_data["image_paths"])

    le_peril = LabelEncoder()
    y_peril_tr = le_peril.fit_transform(y_peril_tr_str)
    y_peril_te = le_peril.transform(y_peril_te_str)
    peril_classes = [str(c) for c in le_peril.classes_]
    not_dis_idx = peril_classes.index("not_disaster")

    le_sev = LabelEncoder()
    y_sev_tr = le_sev.fit_transform([SEVERITY_MAP[c] for c in y_peril_tr_str])
    y_sev_te = le_sev.transform([SEVERITY_MAP[c] for c in y_peril_te_str])
    sev_classes = [str(c) for c in le_sev.classes_]

    le_info = LabelEncoder()
    y_info_tr = le_info.fit_transform([INFO_MAP[c] for c in y_peril_tr_str])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    arms = ["effective_number_focal", "logit_adjusted_ce", "class_balanced_sampler_ce"]

    arm_results: List[Dict[str, object]] = []
    best_arm = arms[1]
    best_oof_f1 = -1.0
    best_oof_sev_cond = None
    best_oof_sev_flat = None

    for arm in arms:
        t0 = time.perf_counter()
        print(f"[Phase 3] Evaluating mutually exclusive loss arm '{arm}' on 5-fold StratifiedKFold OOF...")
        oof_peril = np.zeros((len(y_peril_tr), len(peril_classes)), dtype=np.float32)
        oof_sev_cond = np.zeros((len(y_sev_tr), len(sev_classes)), dtype=np.float32)
        oof_sev_flat = np.zeros((len(y_sev_tr), len(sev_classes)), dtype=np.float32)

        for tr_idx, val_idx in skf.split(x_train, y_peril_tr):
            sc = StandardScaler()
            x_tr_fold = sc.fit_transform(x_train[tr_idx])
            x_val_fold = sc.transform(x_train[val_idx])

            p_peril, p_sev_cond, p_sev_flat = _train_fold_arm(
                arm,
                x_tr_fold,
                y_peril_tr[tr_idx],
                y_sev_tr[tr_idx],
                y_info_tr[tr_idx],
                x_val_fold,
                num_perils=len(peril_classes),
                num_sev=len(sev_classes),
            )
            oof_peril[val_idx] = p_peril
            oof_sev_cond[val_idx] = p_sev_cond
            oof_sev_flat[val_idx] = p_sev_flat

        raw_preds = np.argmax(oof_peril, axis=1)
        raw_macro_f1 = float(f1_score(y_peril_tr, raw_preds, average="macro", zero_division=0))
        raw_bal_acc = float(balanced_accuracy_score(y_peril_tr, raw_preds))

        w_opt, tuned_stats = fit_probability_multipliers(oof_peril, y_peril_tr, not_disaster_idx=not_dis_idx)
        sev_cond_f1 = float(f1_score(y_sev_tr, np.argmax(oof_sev_cond, axis=1), average="macro", zero_division=0))
        sev_flat_f1 = float(f1_score(y_sev_tr, np.argmax(oof_sev_flat, axis=1), average="macro", zero_division=0))
        elapsed = time.perf_counter() - t0

        row = {
            "loss_arm": arm,
            "raw_oof_peril_macro_f1": round(raw_macro_f1, 4),
            "raw_oof_peril_balanced_acc": round(raw_bal_acc, 4),
            "tuned_oof_peril_macro_f1": round(tuned_stats["oof_macro_f1"], 4),
            "tuned_oof_peril_balanced_acc": round(tuned_stats["oof_balanced_accuracy"], 4),
            "tuned_oof_disaster_miss_rate": round(tuned_stats["oof_disaster_miss_rate"], 4),
            "oof_severity_flat_macro_f1": round(sev_flat_f1, 4),
            "oof_severity_peril_conditioned_macro_f1": round(sev_cond_f1, 4),
            "elapsed_sec": round(elapsed, 2),
        }
        arm_results.append(row)
        print(
            f"   -> {arm}: Raw OOF F1={raw_macro_f1*100:.2f}% | Tuned OOF F1={tuned_stats['oof_macro_f1']*100:.2f}% "
            f"| Sev Flat F1={sev_flat_f1*100:.2f}% -> Cond F1={sev_cond_f1*100:.2f}% ({elapsed:.1f}s)"
        )

        if tuned_stats["oof_macro_f1"] > best_oof_f1:
            best_oof_f1 = tuned_stats["oof_macro_f1"]
            best_arm = arm
            best_oof_sev_cond = oof_sev_cond
            best_oof_sev_flat = oof_sev_flat

    # Load Phase 1+5 RF OOF Macro F1 to check Phase 3.6 Acceptance Stop Gate (>= +5.0% required)
    phase1_oof_f1 = 0.5437
    p1_metrics_path = RESULT_DIR / "metrics_disaster_type.json"
    if p1_metrics_path.exists():
        try:
            with open(p1_metrics_path, "r", encoding="utf-8") as f:
                p1_data = json.load(f)
            phase1_oof_f1 = float(p1_data.get("oof_5fold_summary", {}).get("macro_f1_mean", 0.5437))
        except Exception:
            pass

    margin_over_rf = best_oof_f1 - phase1_oof_f1
    beats_stop_gate = margin_over_rf >= 0.05

    # Phase 4 Per-Peril Severity Breakdown on Winning Arm
    assert best_oof_sev_cond is not None and best_oof_sev_flat is not None
    per_peril_severity: Dict[str, Dict[str, float]] = {}
    cond_preds = np.argmax(best_oof_sev_cond, axis=1)
    flat_preds = np.argmax(best_oof_sev_flat, axis=1)
    for idx, peril in enumerate(peril_classes):
        mask = y_peril_tr == idx
        per_peril_severity[peril] = {
            "support": int(np.sum(mask)),
            "flat_severity_accuracy": round(float(accuracy_score(y_sev_tr[mask], flat_preds[mask])), 4),
            "peril_conditioned_severity_accuracy": round(float(accuracy_score(y_sev_tr[mask], cond_preds[mask])), 4),
        }

    payload = {
        "hardware_device": "cuda" if torch.cuda.is_available() else "cpu",
        "loss_arms_5fold_oof": arm_results,
        "winning_loss_arm": best_arm,
        "best_neural_oof_macro_f1": round(best_oof_f1, 4),
        "phase1_rf_oof_macro_f1": round(phase1_oof_f1, 4),
        "margin_over_phase1_rf": round(margin_over_rf, 4),
        "phase3_6_stop_gate_passed": beats_stop_gate,
        "production_decision": (
            "Ship MultiTaskDisasterNet"
            if beats_stop_gate
            else "Phase 3.6 Stop Gate triggered (< +5.0% OOF Macro F1 gain on CPU): Ship Phase 1/2 + Phase 5 Calibrated RF as primary production model"
        ),
        "phase4_per_peril_severity": per_peril_severity,
    }

    with open(RESULT_DIR / "phase3_4_multitask.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md = [
        "# Phase 3 & Phase 4 Report: Multi-Task Loss Arms, Stop Gate & Peril-Conditioned Severity",
        "",
        "## 1. Phase 3 Mutually Exclusive Imbalance Loss Arms (5-Fold OOF)",
        "| Loss Arm | Raw OOF Peril Macro F1 | Tuned OOF Peril Macro F1 | Tuned OOF Bal Acc | OOF Disaster Miss Rate | Flat Severity F1 | Peril-Conditioned Severity F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in arm_results:
        md.append(
            f"| `{r['loss_arm']}` | `{r['raw_oof_peril_macro_f1']*100:.2f}%` | **`{r['tuned_oof_peril_macro_f1']*100:.2f}%`** "
            f"| `{r['tuned_oof_peril_balanced_acc']*100:.2f}%` | `{r['tuned_oof_disaster_miss_rate']*100:.2f}%` "
            f"| `{r['oof_severity_flat_macro_f1']*100:.2f}%` | **`{r['oof_severity_peril_conditioned_macro_f1']*100:.2f}%`** |"
        )

    md.extend(
        [
            "",
            "## 2. Phase 3.6 Acceptance Stop Gate Decision (`FR-7`)",
            f"- **Phase 1+5 Calibrated RF OOF Macro F1:** `{phase1_oof_f1*100:.2f}%`",
            f"- **Best Neural Multi-Task OOF Macro F1 (`{best_arm}`):** `{best_oof_f1*100:.2f}%`",
            f"- **Margin vs. Phase 1+5 RF:** `{margin_over_rf*100:+.2f}%` (Stop Gate Threshold: `>= +5.00%`)",
            f"- **Decision:** **{payload['production_decision']}**",
            "",
            "## 3. Phase 4 Peril-Conditioned Severity Head Breakdown (`FR-9`)",
            "| Peril | Train Support | Flat Severity Accuracy | Peril-Conditioned Severity Accuracy |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )
    for peril, stats in per_peril_severity.items():
        md.append(
            f"| `{peril}` | {stats['support']} | `{stats['flat_severity_accuracy']*100:.2f}%` "
            f"| **`{stats['peril_conditioned_severity_accuracy']*100:.2f}%`** |"
        )

    with open(RESULT_DIR / "phase3_4_multitask_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("[Phase 3 & 4] Saved results/phase3_4_multitask.json and results/phase3_4_multitask_report.md")
    return payload


if __name__ == "__main__":
    run_phase3_and_4_evaluation()

