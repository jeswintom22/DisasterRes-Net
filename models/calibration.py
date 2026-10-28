"""Probability calibration, OOF multiplier search, abstention, and OOD guards.

Implements Phases 1, 5, and 7 of the DisasterRes-Net Classification Improvement Spec:
- Per-class probability multipliers w_k (coordinate search on OOF macro F1)
- Prior-adjustment power scaling (P_k / pi_k^tau)
- Cost-sensitive utility optimization (penalizing disaster -> not_disaster misses)
- Scalar temperature scaling (T > 0) and Expected Calibration Error (ECE)
- Threshold + top-2 margin abstention rule ('uncertain' + top2 classes)
- Mahalanobis + max-softmax out-of-distribution (OOD) guard
- Wilson 95% confidence intervals for small-support classes
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import balanced_accuracy_score, f1_score


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) across probability bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def disaster_miss_rate(y_true: np.ndarray, y_pred: np.ndarray, not_disaster_idx: int) -> float:
    """Fraction of true disaster samples falsely predicted as not_disaster."""
    is_true_disaster = y_true != not_disaster_idx
    total_disasters = int(np.sum(is_true_disaster))
    if total_disasters == 0:
        return 0.0
    missed = int(np.sum(is_true_disaster & (y_pred == not_disaster_idx)))
    return float(missed / total_disasters)


def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Vectorized O(N) Macro-F1 computation via 1D bincount confusion matrix."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0).astype(np.float64) - tp
    fn = np.sum(cm, axis=1).astype(np.float64) - tp
    denom = 2.0 * tp + fp + fn
    f1s = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    return float(np.mean(f1s))


def _fast_macro_f1_and_recalls(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (macro_f1, per_class_recalls, per_class_f1s) in O(N)."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0).astype(np.float64) - tp
    fn = np.sum(cm, axis=1).astype(np.float64) - tp
    support = tp + fn
    recalls = np.where(support > 0, tp / support, 0.0)
    denom = 2.0 * tp + fp + fn
    f1s = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    return float(np.mean(f1s)), recalls, f1s


def apply_multipliers(probs: np.ndarray, multipliers: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply per-class multipliers w_k and temperature T to a probability matrix."""
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0)
    if abs(temperature - 1.0) > 1e-6:
        logits = np.log(clipped) / max(temperature, 1e-4)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        scaled = np.exp(logits)
    else:
        scaled = clipped
    weighted = scaled * np.asarray(multipliers, dtype=np.float64).reshape(1, -1)
    row_sums = np.sum(weighted, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return weighted / row_sums


def fit_prior_correction_tau(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    class_priors: np.ndarray,
    not_disaster_idx: Optional[int] = None,
    disaster_miss_penalty: float = 1.0,
    min_recall_floor: float = 0.32,
) -> Tuple[float, np.ndarray, float]:
    """Fit prior-correction exponent tau in P_k / pi_k^tau on OOF predictions."""
    taus = np.linspace(0.0, 1.5, 61)
    best_tau = 0.0
    best_score = -1e9
    best_w = np.ones_like(class_priors, dtype=np.float64)
    n_classes = len(class_priors)

    for tau in taus:
        w = np.power(np.clip(class_priors, 1e-6, 1.0), -tau)
        # Normalize so minimum weight is 1.0 and bound to [0.5, 20.0]
        w = w / np.min(w)
        w = np.clip(w, 0.5, 20.0)
        if not_disaster_idx is not None and disaster_miss_penalty > 1.0:
            w[not_disaster_idx] = max(0.5, w[not_disaster_idx] / disaster_miss_penalty)

        adj_probs = apply_multipliers(oof_probs, w)
        preds = np.argmax(adj_probs, axis=1)
        macro_f1, recalls, _ = _fast_macro_f1_and_recalls(y_true, preds, n_classes)
        recall_shortfall = float(np.sum(np.maximum(0.0, min_recall_floor - recalls)))
        bal_acc = float(np.mean(recalls))
        if not_disaster_idx is not None and disaster_miss_penalty > 1.0:
            miss = disaster_miss_rate(y_true, preds, not_disaster_idx)
            score = macro_f1 + 0.15 * bal_acc - 0.35 * recall_shortfall - 0.15 * (disaster_miss_penalty - 1.0) * miss
        else:
            score = macro_f1 + 0.15 * bal_acc - 0.35 * recall_shortfall

        if score > best_score:
            best_score = score
            best_tau = float(tau)
            best_w = w.copy()

    return best_tau, best_w, best_score


def fit_probability_multipliers(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    not_disaster_idx: Optional[int] = None,
    disaster_miss_penalty: float = 1.0,
    bounds: Tuple[float, float] = (0.5, 20.0),
    n_rounds: int = 4,
    min_recall_floor: float = 0.32,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Coordinate search for per-class probability multipliers w_k in [0.5, 20].

    Initializes from the best prior-correction exponent tau, then refines each
    class multiplier on a logarithmic grid to maximize OOF Macro F1 + balanced
    recall floor (with optional cost-sensitive penalty on disaster -> not_disaster misses).
    """
    n_classes = oof_probs.shape[1]
    counts = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    priors = counts / max(1.0, np.sum(counts))

    best_tau, w, _ = fit_prior_correction_tau(
        oof_probs,
        y_true,
        priors,
        not_disaster_idx=not_disaster_idx,
        disaster_miss_penalty=disaster_miss_penalty,
        min_recall_floor=min_recall_floor,
    )

    def _objective(candidate_w: np.ndarray) -> float:
        adj = apply_multipliers(oof_probs, candidate_w)
        preds = np.argmax(adj, axis=1)
        macro_f1, recalls, _ = _fast_macro_f1_and_recalls(y_true, preds, n_classes)
        recall_shortfall = float(np.sum(np.maximum(0.0, min_recall_floor - recalls)))
        bal_acc = float(np.mean(recalls))
        base = macro_f1 + 0.15 * bal_acc - 0.35 * recall_shortfall
        if not_disaster_idx is not None and disaster_miss_penalty > 1.0:
            miss = disaster_miss_rate(y_true, preds, not_disaster_idx)
            return base - 0.15 * (disaster_miss_penalty - 1.0) * miss
        return base

    best_obj = _objective(w)
    grid = np.unique(np.concatenate([np.geomspace(bounds[0], bounds[1], 45), [1.0]]))

    # Coordinate search ordered from rarest class to most frequent class
    class_order = np.argsort(counts)
    for _ in range(n_rounds):
        improved = False
        for k in class_order:
            current_best_wk = w[k]
            for cand in grid:
                w[k] = cand
                val = _objective(w)
                if val > best_obj + 1e-5:
                    best_obj = val
                    current_best_wk = cand
                    improved = True
            w[k] = current_best_wk
        if not improved:
            break

    adj = apply_multipliers(oof_probs, w)
    preds = np.argmax(adj, axis=1)
    metrics = {
        "init_tau": best_tau,
        "oof_macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "oof_accuracy": float(np.mean(preds == y_true)),
        "oof_disaster_miss_rate": (
            disaster_miss_rate(y_true, preds, not_disaster_idx) if not_disaster_idx is not None else 0.0
        ),
    }
    return w, metrics


def fit_temperature_scaling(oof_probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
    """Fit scalar temperature T > 0 on OOF probabilities to minimize NLL and ECE."""
    eps = 1e-12
    log_probs = np.log(np.clip(oof_probs, eps, 1.0))
    ece_before = expected_calibration_error(oof_probs, y_true)

    def _nll(temp: float) -> float:
        scaled = log_probs / max(temp, 1e-3)
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        p = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        chosen = p[np.arange(len(y_true)), y_true]
        return float(-np.mean(np.log(np.clip(chosen, eps, 1.0))))

    res = minimize_scalar(_nll, bounds=(0.15, 5.0), method="bounded")
    best_t = float(res.x) if res.success else 1.0

    cal_probs = apply_multipliers(oof_probs, np.ones(oof_probs.shape[1]), temperature=best_t)
    ece_after = expected_calibration_error(cal_probs, y_true)
    return best_t, ece_before, ece_after


@dataclass
class AbstentionConfig:
    """Thresholds for emitting 'uncertain' with top-2 candidate classes."""

    min_confidence: float = 0.38
    min_top2_margin: float = 0.06
    per_class_thresholds: Dict[str, float] = field(default_factory=dict)
    ood_mahalanobis_threshold: float = 45.0


def tune_abstention_thresholds(
    calibrated_probs: np.ndarray,
    y_true: np.ndarray,
    classes: Sequence[str],
    max_abstain_rate: float = 0.15,
    target_precision: float = 0.80,
) -> Tuple[AbstentionConfig, Dict[str, float]]:
    """Tune confidence and top-2 margin thresholds on OOF calibrated probabilities."""
    sorted_p = np.sort(calibrated_probs, axis=1)
    max_p = sorted_p[:, -1]
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    preds = np.argmax(calibrated_probs, axis=1)
    correct = preds == y_true

    best_cfg = AbstentionConfig(min_confidence=0.35, min_top2_margin=0.05)
    best_score = -1.0
    best_stats: Dict[str, float] = {}

    for conf_th in np.linspace(0.22, 0.58, 19):
        for marg_th in np.linspace(0.02, 0.16, 15):
            abstain = (max_p < conf_th) | (margin < marg_th)
            abstain_rate = float(np.mean(abstain))
            if abstain_rate > max_abstain_rate or abstain_rate >= 0.95:
                continue
            answered = ~abstain
            answered_acc = float(np.mean(correct[answered]))
            answered_macro_f1 = float(f1_score(y_true[answered], preds[answered], average="macro", zero_division=0))
            # Reward high answered Macro F1 and accuracy while keeping abstention within budget
            score = answered_macro_f1 + 0.5 * answered_acc - 0.15 * abstain_rate
            if answered_acc >= target_precision:
                score += 0.05
            if score > best_score:
                best_score = score
                best_cfg = AbstentionConfig(
                    min_confidence=round(float(conf_th), 4),
                    min_top2_margin=round(float(marg_th), 4),
                    per_class_thresholds={str(c): round(float(conf_th), 4) for c in classes},
                )
                best_stats = {
                    "abstain_rate": round(abstain_rate, 4),
                    "answered_accuracy": round(answered_acc, 4),
                    "answered_macro_f1": round(answered_macro_f1, 4),
                }

    return best_cfg, best_stats


class MahalanobisOODGuard:
    """Lightweight diagonal-shrinkage Mahalanobis OOD detector on CNN embeddings."""

    def __init__(self, percentile_threshold: float = 98.5) -> None:
        self.percentile_threshold = percentile_threshold
        self.centroids_: Optional[np.ndarray] = None
        self.inv_var_: Optional[np.ndarray] = None
        self.threshold_: float = 100.0

    def fit(self, embeddings: np.ndarray, y_true: np.ndarray) -> "MahalanobisOODGuard":
        x = np.asarray(embeddings, dtype=np.float32)
        classes = np.unique(y_true)
        centroids = []
        for c in classes:
            centroids.append(np.mean(x[y_true == c], axis=0))
        self.centroids_ = np.vstack(centroids)
        var = np.var(x, axis=0) + 1e-2
        self.inv_var_ = 1.0 / var
        scores = self.score(x)
        self.threshold_ = float(np.percentile(scores, self.percentile_threshold))
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self.centroids_ is None or self.inv_var_ is None:
            return np.zeros(len(embeddings), dtype=np.float32)
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Normalized Mahalanobis distance to nearest class centroid
        diffs = x[:, None, :] - self.centroids_[None, :, :]
        dists = np.sqrt(np.mean((diffs * diffs) * self.inv_var_[None, None, :], axis=2))
        return np.min(dists, axis=1)

    def is_ood(self, embeddings: np.ndarray, max_prob: Optional[np.ndarray] = None) -> np.ndarray:
        dists = self.score(embeddings)
        flag = dists > self.threshold_
        if max_prob is not None:
            flag = flag | (np.asarray(max_prob) < 0.18)
        return flag


@dataclass
class CalibratedModelBundle:
    """Unified artifact wrapper storing multipliers, temperature, abstention, and OOD state."""

    objective: str
    feature_pipeline: str
    classes: List[str]
    multipliers: List[float]
    temperature: float
    abstention: AbstentionConfig
    operating_point: str = "macro_f1_optimal"
    cost_sensitive_multipliers: Optional[Dict[str, List[float]]] = None
    ood_guard: Optional[MahalanobisOODGuard] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def evaluate_probabilities(
        self,
        raw_probs: np.ndarray,
        cnn_embedding: Optional[np.ndarray] = None,
        operating_mode: str = "default",
    ) -> Dict[str, object]:
        """Convert raw model probabilities (1D or 2D row) into calibrated decision output."""
        p = np.asarray(raw_probs, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(1, -1)

        mult = self.multipliers
        if (
            operating_mode != "default"
            and self.cost_sensitive_multipliers
            and operating_mode in self.cost_sensitive_multipliers
        ):
            mult = self.cost_sensitive_multipliers[operating_mode]

        cal_p = apply_multipliers(p, np.asarray(mult, dtype=np.float64), temperature=self.temperature)[0]
        order = np.argsort(cal_p)[::-1]
        top1_idx, top2_idx = int(order[0]), int(order[1])
        top1_cls, top2_cls = self.classes[top1_idx], self.classes[top2_idx]
        top1_prob, top2_prob = float(cal_p[top1_idx]), float(cal_p[top2_idx])
        margin = top1_prob - top2_prob

        cls_thresh = self.abstention.per_class_thresholds.get(top1_cls, self.abstention.min_confidence)
        uncertain = (top1_prob < cls_thresh) or (margin < self.abstention.min_top2_margin)

        ood_flag = False
        ood_score = 0.0
        if self.ood_guard is not None and cnn_embedding is not None:
            ood_score = float(self.ood_guard.score(cnn_embedding)[0])
            ood_flag = bool(self.ood_guard.is_ood(cnn_embedding, np.array([top1_prob]))[0])
            if ood_flag:
                uncertain = True

        effective_pred = "uncertain" if uncertain else top1_cls
        return {
            "prediction": top1_cls,
            "effective_prediction": effective_pred,
            "uncertain": bool(uncertain),
            "confidence": top1_prob,
            "raw_confidence": float(np.max(p[0])),
            "top2_margin": round(margin, 4),
            "top2": [
                {"label": top1_cls, "confidence": round(top1_prob * 100.0, 2)},
                {"label": top2_cls, "confidence": round(top2_prob * 100.0, 2)},
            ],
            "probabilities": {cls: float(prob) for cls, prob in zip(self.classes, cal_p)},
            "raw_probabilities": {cls: float(prob) for cls, prob in zip(self.classes, p[0])},
            "ood_flag": ood_flag,
            "ood_score": round(ood_score, 4),
            "calibration_temperature": round(self.temperature, 4),
        }
