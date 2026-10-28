# Implementation Notes: Classification Improvement

**Feature ID:** `003-classification-improvement`

---

## 1. Key Codebase Touchpoints

| File | Current Role | Planned Modifications |
| :--- | :--- | :--- |
| `step3_classify.py` | Extracts 1,000-D CNN logits + GLCM + LBP, scales on full train split, fits RF. | Use `timm.data.resolve_data_config` (`0.5/0.5` norm), support `num_classes=0` (1,536-D pooled), wrap `StandardScaler` + `RF` in `Pipeline`, log OOF `f1_macro` & `balanced_accuracy`. |
| `fusion/feature_fusion.py` | Defines `df1_df2_glcm_lbp` (2030-D) and `df1_df2_glcm_lbp_stats` (2036-D) specs. | Register `v3` feature specs (1,536-D pooled ± `mixed_6a` 1,088-D ± GLCM/LBP) and validate calibration/multiplier metadata in `resolve_artifact_spec`. |
| `models/hybrid_pipeline.py` | Runs LMS saliency + LBP + GLCM + CNN and calls `clf.predict()`. | Support pruned stream specs (skipping LMS saliency if dropped in Phase 2), apply OOF probability multipliers $\mathbf{w}$, temperature scaling $T$, `'uncertain'` abstention, and OOD guard. |
| `damage_assessment/localization.py` | Computes Module M2 damage mask, DDM, and DEM score. | Accept calibrated severity confidence in DEM calculation; log DEM version to track distribution shift. |
| `crew/pipeline_singleton.py` | Manages singleton pipeline and 2-worker `INFERENCE_POOL`. | Add `DISASTERRES_MODEL_BACKEND` config flag to toggle between `legacy_rf`, `calibrated_rf`, and `multitask_nn` for zero-downtime rollback. |
| `app.py` | Exposes `/api/predict` and `/api/chat`. | Add `calibrated_probabilities`, `uncertain`, `top2`, and `ood_flag` to `/api/predict` response while keeping legacy keys intact for `templates/index.html`. |
| `crew/report/facts.py` & `crew/crew.py` | Packages `ScoutResult` and `AnalysisResult` into `FactsBlock`. | Add `uncertain` and `calibrated_confidence` fields; when `uncertain=True`, prompt Analyst/Scout to request corroborating imagery or report top-2 candidates. |

---

## 2. TIMM `inception_resnet_v2` Layer & Config Reference

- **Pretrained Data Config Verification:**
  ```python
  import timm
  from timm.data import resolve_data_config, create_transform

  model = timm.create_model("inception_resnet_v2", pretrained=True, num_classes=0)
  config = resolve_data_config({}, model=model)
  # Expected: {'input_size': (3, 299, 299), 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), ...}
  transform_eval = create_transform(**config, is_training=False)
  ```
- **Stage B & Stage C Unfreezing Layer Names:**
  Verify via `[name for name, _ in model.named_children()]`:
  - **Stage B (Top blocks):** `mixed_7a`, `repeat_2` (9 `Block8` modules), `block8`, `conv2d_7b`.
  - **Stage C (Mid-high blocks):** `mixed_6a`, `repeat_1[10:20]` (upper 10 of the 20 `Block17` modules).
- **BatchNorm Eval Hook for Small Physical Batch Sizes ($<32$):**
  ```python
  def freeze_bn_stats(module: torch.nn.Module) -> None:
      for m in module.modules():
          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
              m.eval()
  ```
  Must be called inside `MultiTaskDisasterNet.train(mode=True)` whenever physical batch size per device is $<32$.

---

## 3. Out-of-Fold (OOF) Coordinate Search & Prior Correction

- **Prior Correction Formula (1-parameter baseline):**
  $$\tilde{P}_k(x) \propto \frac{P_k(x)}{\pi_k^\tau}, \quad \tau \in [0.2, 1.5]$$
  where $\pi_k = n_k / N_{\text{train}}$ is the empirical training class prior.
- **Coordinate Search (6-parameter refinement):**
  Initialize $w_k = \pi_k^{-\tau^*}$, then iteratively optimize each $w_k \in [0.5, 20.0]$ on a logarithmic grid to maximize OOF `f1_macro` (or cost-weighted utility where `disaster` $\rightarrow$ `not_disaster` carries penalty $\lambda \in \{2.0, 3.0\}$).

---

## 4. Confidence Intervals for Small Support Classes

Because `landslide` ($n=28$ in test) and `flood` ($n=96$ in test) have small sample sizes, point estimates carry wide uncertainty:
- For `landslide` ($n=28$), a single image changes recall by $3.57\%$.
- Always compute **Wilson score 95% confidence intervals** for per-class recall and precision on classes with $<100$ samples, and report fold-level standard deviation across the 5 OOF splits.
