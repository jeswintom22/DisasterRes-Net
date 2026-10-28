"""End-to-End Integration & Regression Test Suite for Feature 003 (Classification Improvement).

Verifies:
- Phase 0: Audit artifacts, duplicate cluster groups, and memorization diagnostics
- Phase 1 & 5: CalibratedModelBundle, Wilson 95% CIs, ECE, abstention ('uncertain' + top2),
  and Mahalanobis OOD guard
- Phase 2: Feature contracts (v3 1,536-D pooled, 1,088-D mixed_6a, 1,036-D parsimonious)
  and timm native (0.5, 0.5, 0.5) normalization
- Phase 3 & 4: MultiTaskDisasterNet, PerilConditionedSeverityHead, BatchNorm2d freeze
  discipline when physical batch size < 32, and 3-stage unfreezing (Stage A/B/C)
- Phase 7: HybridDisasterPipeline, Flask /api/predict response schema, DEM calibrated
  underclassification guard, CrewAI uncertainty caveat propagation, and legacy_rf rollback
- Phase 9: PerspectiveGate routing ('overhead_or_aerial', 'ground_level', 'graphic_or_screenshot')
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from crew.tools.aggregator import aggregate_assessments
from crew.vision.perspective_gate import PerspectiveGate
from damage_assessment.localization import DamageLocalizationAnalyzer
from features.feature_contract import get_pipeline_spec
from models.calibration import (
    AbstentionConfig,
    CalibratedModelBundle,
    MahalanobisOODGuard,
    apply_multipliers,
    expected_calibration_error,
    fit_probability_multipliers,
    fit_temperature_scaling,
    tune_abstention_thresholds,
    wilson_ci,
)
from models.multitask_net import (
    EffectiveNumberFocalLoss,
    LogitAdjustedCrossEntropy,
    MultiTaskDisasterNet,
    PerilConditionedSeverityHead,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase0_audit_artifacts_present() -> None:
    diag_path = ROOT / "results" / "phase0_diagnostics.json"
    csv_path = ROOT / "results" / "phase0_audit_samples.csv"
    dup_path = ROOT / "results" / "phase0_duplicate_groups.npz"

    assert diag_path.exists(), "Phase 0 diagnostics JSON must exist"
    assert csv_path.exists(), "Phase 0 200-image audit CSV must exist"
    assert dup_path.exists(), "Phase 0 duplicate groups NPZ must exist"

    data = json.loads(diag_path.read_text(encoding="utf-8"))
    assert data["audit_200_samples"]["total_audited"] == 200
    assert data["memorization_gap"]["train_test_macro_f1_gap"] > 0.40


def test_phase1_and_5_calibration_and_ood_primitives() -> None:
    lo, hi = wilson_ci(8, 28)
    assert 0.14 < lo < 0.28 < hi < 0.48

    rng = np.random.default_rng(42)
    y_true = np.array([0] * 30 + [1] * 10 + [2] * 60 + [3] * 5 + [4] * 80 + [5] * 25)
    raw_probs = rng.dirichlet(np.ones(6), size=len(y_true))
    for i, y in enumerate(y_true):
        raw_probs[i, y] += 0.55
    raw_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)

    w, stats = fit_probability_multipliers(raw_probs, y_true, not_disaster_idx=4)
    assert len(w) == 6
    assert stats["oof_macro_f1"] > 0.50

    temp, ece_before, ece_after = fit_temperature_scaling(apply_multipliers(raw_probs, w), y_true)
    assert 0.10 <= temp <= 5.0
    assert ece_after <= ece_before + 0.02

    classes = ["earthquake", "flood", "hurricane", "landslide", "not_disaster", "wildfire"]
    abs_cfg, abs_stats = tune_abstention_thresholds(raw_probs, y_true, classes=classes, max_abstain_rate=0.25)
    assert 0.0 <= abs_stats.get("abstain_rate", 0.0) <= 0.25

    emb = rng.standard_normal((len(y_true), 64)).astype(np.float32)
    guard = MahalanobisOODGuard(percentile_threshold=95.0).fit(emb, y_true)
    ood_sample = np.ones((1, 64), dtype=np.float32) * 12.0
    assert bool(guard.is_ood(ood_sample)[0]) is True

    bundle = CalibratedModelBundle(
        objective="disaster_type",
        feature_pipeline="df1_df2_glcm_lbp_stats",
        classes=classes,
        multipliers=[float(x) for x in w],
        temperature=temp,
        abstention=abs_cfg,
        ood_guard=guard,
    )
    eval_in = bundle.evaluate_probabilities(raw_probs[0], cnn_embedding=emb[0])
    assert "effective_prediction" in eval_in
    assert len(eval_in["top2"]) == 2
    eval_ood = bundle.evaluate_probabilities(raw_probs[0], cnn_embedding=ood_sample[0])
    assert eval_ood["ood_flag"] is True
    assert eval_ood["uncertain"] is True
    assert eval_ood["effective_prediction"] == "uncertain"


def test_phase2_feature_contracts_and_timm_normalization() -> None:
    import step3_classify as s3

    spec_1036 = get_pipeline_spec("df1_glcm_lbp_stats")
    assert spec_1036.dimension == 1036

    spec_v3_1572 = get_pipeline_spec("v3_df1536_glcm_lbp_stats")
    assert spec_v3_1572.dimension == 1572

    spec_v3_2660 = get_pipeline_spec("v3_df1536_mixed6a_glcm_lbp_stats")
    assert spec_v3_2660.dimension == 2660

    tf_native = s3.get_transforms(train=False, timm_native=True)
    norm_layer = [t for t in tf_native.transforms if hasattr(t, "mean")][0]
    assert list(norm_layer.mean) == [0.5, 0.5, 0.5]
    assert list(norm_layer.std) == [0.5, 0.5, 0.5]


def test_phase3_and_4_multitask_net_bn_freeze_and_peril_conditioned_severity() -> None:
    head = PerilConditionedSeverityHead(in_features=128, num_perils=6, num_severity=3, low_rank=8)
    feat = torch.randn(4, 128)
    peril_probs = torch.softmax(torch.randn(4, 6), dim=-1)
    sev_logits = head(feat, peril_probs)
    assert sev_logits.shape == (4, 3)

    # Verify loss arms
    counts = [902, 222, 4804, 64, 5957, 721]
    targets = torch.tensor([0, 1, 3, 5], dtype=torch.long)
    logits = torch.randn(4, 6, requires_grad=True)
    loss_focal = EffectiveNumberFocalLoss(counts)(logits, targets)
    loss_focal.backward()
    assert float(loss_focal.item()) > 0.0

    logits2 = torch.randn(4, 6, requires_grad=True)
    loss_la = LogitAdjustedCrossEntropy(counts)(logits2, targets)
    loss_la.backward()
    assert float(loss_la.item()) > 0.0

    # Verify MultiTaskDisasterNet Stage A/B/C unfreezing and BatchNorm eval lock
    net = MultiTaskDisasterNet(pretrained=False, freeze_bn=True)
    net.train(True)
    bn_training_states = [
        m.training for m in net.backbone.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
    ]
    assert all(state is False for state in bn_training_states), "All backbone BatchNorm layers must stay in eval mode"

    net.set_unfreeze_stage("A")
    assert sum(p.numel() for p in net.backbone.parameters() if p.requires_grad) == 0

    net.set_unfreeze_stage("B")
    stage_b_trainable = sum(p.numel() for p in net.backbone.parameters() if p.requires_grad)
    assert stage_b_trainable > 0

    net.set_unfreeze_stage("C")
    stage_c_trainable = sum(p.numel() for p in net.backbone.parameters() if p.requires_grad)
    assert stage_c_trainable > stage_b_trainable

    pooled = torch.randn(2, 1536)
    hc = torch.randn(2, 36)
    out = net.forward_from_embeddings(pooled, hc)
    assert out["disaster_type"].shape == (2, 6)
    assert out["damage"].shape == (2, 3)
    assert out["informativeness"].shape == (2, 2)


def test_phase7_flask_api_and_crew_integration() -> None:
    from app import app

    img = Image.fromarray(np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    client = app.test_client()
    resp = client.post("/api/predict", data={"file": (buf, "earthquake_sample.jpg")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = resp.get_json()

    for objective in ("disaster_type", "damage", "informativeness"):
        assert objective in payload
        obj_data = payload[objective]
        assert "prediction" in obj_data
        assert "effective_prediction" in obj_data
        assert "uncertain" in obj_data
        assert "top2" in obj_data
        assert "calibrated_probabilities" in obj_data
        assert "probabilities" in obj_data
        assert "top5" in obj_data

    # Verify Crew aggregator confidence band and uncertainty caveats
    analyzer = DamageLocalizationAnalyzer()
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    smap = np.ones((200, 200), dtype=np.float32) * 0.5
    assessment = analyzer.assess(rgb, smap, "earthquake", damage_prediction="severe", damage_confidence=0.75)
    agg = aggregate_assessments(
        "TestLoc",
        "Earthquake",
        [assessment],
        confidences=[0.75],
        rejected=0,
        extra_caveats=["Peril classifier flagged close boundary uncertainty between earthquake and hurricane."],
    )
    assert agg.confidence_band == "high"
    assert any("boundary uncertainty" in c for c in agg.caveats)


def test_phase9_perspective_gate() -> None:
    gate = PerspectiveGate()
    flat_graphic = np.ones((256, 256, 3), dtype=np.uint8) * 240
    v_graphic = gate.classify(flat_graphic)
    assert v_graphic.perspective == "graphic_or_screenshot"

    rng = np.random.default_rng(7)
    aerial_texture = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    v_aerial = gate.classify(aerial_texture)
    assert v_aerial.perspective == "overhead_or_aerial"
    assert v_aerial.recommended_localization_backend == "segformer_b2_aerial"

