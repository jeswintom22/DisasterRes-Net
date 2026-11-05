# Phased Implementation Plan: DisasterRes-Net Classification Improvement

**Feature ID:** `003-classification-improvement`  
**Spec Reference:** `project-delivery/features/003-classification-improvement/spec.md`  
**Source Blueprint:** `DisasterRes-Net Classification Improvement Spec.pdf` (Parallax Machine Inc.)

---

## 1. Architecture & Execution Dependency Graph

```
[Phase 0: Diagnostics, Duplicate & 200-Image Label Audit]
   │
   ├──► [Phase 1: RF Regularization & OOF Probability Multiplier Tuning] (Days, CPU-only, cached .npz)
   │       │
   │       └──► [Phase 2: TIMM Normalization (0.5/0.5), 1536-D Pooled Features & Stream Ablation]
   │               │
   │               ├──► [Phase 6: Rare-Class Data Expansion (>=300/class, source-disjoint)] (Parallel)
   │               │
   │               └──► [Phase 3: End-to-End 3-Stage Multi-Task Fine-Tuning (GPU)]
   │                       │
   │                       ├──► [Phase 4: Soft Peril-Conditioned Severity Head] (Only if Peril Recall >= 50%)
   │                       │
   │                       └──► [Phase 5: Temperature Calibration, Abstention ('uncertain'), TTA & OOD Guard]
   │                               │
   │                               └──► [Phase 7: Artifact Bundle, Flask /api/predict, DEM & CrewAI Integration]
   │                                       │
   │                                       └──► [Phase 8: Comparison Matrix & Release Gates]
   │                                               │
   │                                               └──► [Phase 9 (Deferred): SegFormer B3 Building Damage]
```

### Critical Path & Risk Summary
- **Critical Path:** `Phase 0` $\rightarrow$ `Phase 1` $\rightarrow$ `Phase 2` $\rightarrow$ `Phase 3` $\rightarrow$ `Phase 5` $\rightarrow$ `Phase 7` $\rightarrow$ `Phase 8`.
- **Fastest High-ROI Milestone:** **Phase 1** (recovers rare-class recall in hours on CPU using existing cached 2,036-D `.npz` matrices without re-extracting CNN features).
- **Riskiest Phase:** **Phase 3 (End-to-End Fine-Tuning)** — unfreezing a 56M-parameter `inception_resnet_v2` backbone on imbalanced data (28 landslides vs. 2,060 hurricanes) can overfit rapidly or destabilize `BatchNorm2d` running statistics. Mitigated by an explicit stop gate: if Phase 3 does not beat Phase 1/2 OOF Macro F1 by $\ge 5.0$ points, ship the Phase 1/2 model.
- **Project-Specific Constraints (from `learnings.md`):**
  1. **Windows `spawn` Multiprocessing:** Never initialize `torch.device("cuda")` at module top-level; keep CUDA initialization lazy inside worker functions.
  2. **OpenBLAS Contention:** Enforce `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and cap worker pools at `min(8, os.cpu_count())`.
  3. **Feature Contract Integrity:** `fusion/feature_fusion.py` (`resolve_artifact_spec`) must continue loading legacy 2,030-D and 2,036-D Random Forest bundles alongside new `v3` (1,536-D pooled) bundles.

---

## 2. Phase Breakdown

### Phase 0: Diagnostics and Data Audit
- **Goal:** Quantify how much of the 31.5% disaster miss rate and 62% `hurricane` $\leftrightarrow$ `not_disaster` error is model bias vs. cross-split duplicate leakage and label noise.
- **Risk:** Low (read-only analysis, no model changes).
- **Dependencies:** Existing `.npz` feature caches and `split_dataset/{train,test}`.
- **Files to Create / Modify:**
  - Create `evaluation/phase0_diagnostics.py`
  - Create `results/phase0_audit_report.md` and `results/phase0_audit_samples.csv`
- **Implementation Steps:**
  1. Compute `clf.score(x_train, y_train)` vs. OOF (`0.6806`) and test (`0.6845`) to confirm tree memorization.
  2. Tabulate train/test counts per peril and per severity, logging imbalance ratios (e.g., 73.6:1 `hurricane` to `landslide`).
  3. Compute `pHash`/`dHash` (via `PIL` / `imagehash`) and L2-normalized `df1` cosine similarity across all 12,670 train and 5,433 test images; output duplicate clusters crossing train/test splits and generate `GroupKFold` group IDs.
  4. Sample 50 images each from the 4 largest error cells (`hurricane` $\rightarrow$ `not_disaster` [643], `not_disaster` $\rightarrow$ `hurricane` [419], `earthquake` $\rightarrow$ `hurricane` [183], `wildfire` $\rightarrow$ `hurricane` [111]) into an audit sheet with columns `model_wrong | label_wrong | ambiguous` and `overhead | ground_level | graphic`.
- **Anti-Patterns to Avoid:**
  - Do not clean or relabel the test set based on model predictions without human verification.
- **Verification Strategy:**
  - Execute `python -m evaluation.phase0_diagnostics` and verify `results/phase0_audit_report.md` contains memorization gap, duplicate cluster count, and 200-sample audit manifest. If label noise $>20\%$, insert a label-cleaning pass on training data before Phase 3.

---

### Phase 1: Fix the Decision Rule on the Existing Random Forest
- **Goal:** Recover rare-class recall (`flood`, `landslide`, `earthquake`, `wildfire`) and raise Macro F1 from 40.2% to $\ge 55.0\%$ without re-extracting CNN features.
- **Risk:** Medium — coordinate search on 28-sample `landslide` can overfit OOF if multipliers are unbounded.
- **Dependencies:** Phase 0 duplicate cluster groups (if leakage was found); cached 2,036-D `.npz` matrices.
- **Files to Create / Modify:**
  - Create `evaluation/oof_tuning.py` (in-fold `Pipeline`, coordinate search for $\mathbf{w}$, prior adjustment $\tau$, cost-sensitive matrix)
  - Modify `step3_classify.py` (wrap `StandardScaler` + `RandomForestClassifier` in `sklearn.pipeline.Pipeline`, add `balanced_subsample` and `min_samples_leaf` grid, log OOF `f1_macro` and `balanced_accuracy`)
  - Modify `models/hybrid_pipeline.py` (apply learned probability multipliers $\mathbf{w}$ before `argmax`)
- **Implementation Steps:**
  1. Build an `sklearn.pipeline.Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(...))])` so scaling is fitted strictly inside each fold.
  2. Grid search over `class_weight="balanced_subsample"`, `min_samples_leaf` $\in \{1, 3, 5, 10\}$, `max_features` $\in \{\text{"sqrt"}, 0.1\}$, and `n_estimators` $\in \{300, 500\}$ using 5-fold `cross_val_predict(..., method="predict_proba")` on the 12,670 training images.
  3. Fit per-class probability multipliers $\mathbf{w} \in [0.5, 20.0]^6$ ($\hat{y} = \arg\max_k (P_k \cdot w_k)$) via cyclic coordinate ascent maximizing OOF Macro F1, and compare against prior correction $P_k / f_k^\tau$ ($\tau \in [0.2, 1.5]$).
  4. Evaluate a cost-sensitive decision rule penalizing `disaster` $\rightarrow$ `not_disaster` by $2\times$ and $3\times$; record both operating points.
  5. Freeze the winning RF + $\mathbf{w}$ bundle into `saved_models/` and evaluate **once** on the 5,433-image test set.
- **Anti-Patterns to Avoid:**
  - Never tune $\mathbf{w}$ or `min_samples_leaf` on the 5,433-image test set.
  - Never fit `StandardScaler` on the full 12,670 training set before `cross_val_predict`.
- **Verification Strategy:**
  - Verify OOF and test Macro F1 increase by $\ge 10$ points over 40.2% (target $\ge 55\%$), `flood` and `landslide` recall reach $\ge 30\%$, and no class recall drops $>5$ points below baseline.

---

### Phase 2: Feature Upgrade and Stream Ablation
- **Goal:** Fix the `timm` `inception_resnet_v2` normalization bug (`0.5/0.5` vs. ImageNet mean/std), upgrade from 1,000-D ImageNet logits to 1,536-D pooled embeddings, and prune redundant feature streams.
- **Risk:** Medium — changing feature dimensionality requires careful versioning so existing 2,030-D and 2,036-D checkpoints do not break.
- **Dependencies:** Phase 1 regularizer and OOF evaluation harness.
- **Files to Create / Modify:**
  - Modify `step3_classify.py` (use `timm.data.resolve_data_config` and `create_transform`, instantiate `inception_resnet_v2` with `num_classes=0` for 1,536-D pooled output + optional forward hook on `mixed_6a` for 1,088-D pooled texture)
  - Modify `fusion/feature_fusion.py` (register `v3` feature specs in `PIPELINE_SPECS` and `resolve_artifact_spec`)
  - Modify `evaluation/ablation.py` (run the full OOF Macro F1 ablation grid)
- **Implementation Steps:**
  1. Replace hardcoded `IMAGENET_MEAN` / `IMAGENET_STD` in `step3_classify.py` with `timm.data.resolve_data_config({}, model=model)` (`mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)`, `299x299`).
  2. Extract and cache versioned `.npz` matrices (`{objective}_{split}_v3_df1536_..._features.npz`) with 1,536-D pooled features and 1,088-D `mixed_6a` pooled features.
  3. Run the OOF Macro F1 ablation grid using Phase 1's best RF settings across:
     - Streams: `df1` only; `df1 + df2`; `df1 + glcm + lbp`; `df1 + df2 + glcm + lbp`.
     - Representations: 1,000-D logits vs. 1,536-D pooled vs. 1,536-D + 1,088-D `mixed_6a`.
  4. Apply the parsimony rule: select the cheapest feature set within $1.0$ OOF Macro F1 point of the best. If `df2` (saliency), `glcm`, or `lbp` add $< 1.0$ point, drop them from the active spec to eliminate LMS saliency compute overhead at inference time.
- **Anti-Patterns to Avoid:**
  - Do not overwrite legacy `.npz` or `.joblib` files; version all filenames with the `v3` spec name.
- **Verification Strategy:**
  - Confirm MLflow logs the complete ablation table, unit tests verify legacy 2,030-D/2,036-D checkpoints still load via `resolve_artifact_spec`, and inference latency improves if saliency/GLCM/LBP are pruned.

---

### Phase 3: End-to-End Multi-Task Fine-Tuning
- **Goal:** Train a shared `inception_resnet_v2` multi-task network (`peril` [6], `damage` [3], `informativeness` [2]) to push Macro F1 $\ge 70\%$, all per-class recalls $\ge 50\%$, and disaster-to-`not_disaster` miss rate $\le 15\%$.
- **Risk:** High — GPU memory limits, BatchNorm corruption on small batch sizes, and overfitting minority perils.
- **Dependencies:** Phase 2 winning feature/stream configuration; Phase 0 label cleanup (if noise $>20\%$).
- **Files to Create / Modify:**
  - Create `models/multitask_net.py` (`MultiTaskDisasterNet`, effective-number Focal Loss, Logit-Adjusted CE, BatchNorm freeze hook)
  - Create `train_multitask.py` (3-stage unfreezing loop, 5-fold CV, early stopping on OOF Macro F1, AMP, gradient accumulation)
- **Implementation Steps:**
  1. Build `MultiTaskDisasterNet`: shared `inception_resnet_v2` (`num_classes=0`, 1,536-D) + optional concatenation of surviving handcrafted features $\rightarrow$ three heads (`Dropout(0.4)` + `Linear`), label smoothing $0.05$.
  2. Implement and benchmark the 3 mutually exclusive imbalance arms on Fold 1 before full 5-fold runs:
     - Arm (a): Focal Loss ($\gamma=2$) + effective-number weights ($\beta=0.999$).
     - Arm (b): Logit-Adjusted CE ($z_k + \log \pi_k$).
     - Arm (c): Class-Balanced Sampler + plain CE.
  3. Multi-task loss: $L = L_{\text{peril}} + 0.5 L_{\text{damage}} + 0.25 L_{\text{info}}$ with binary masking for missing labels.
  4. Execute the 3-stage unfreezing schedule:
     - **Stage A (3–5 epochs):** Freeze backbone, train heads at `lr=1e-3`.
     - **Stage B (5–10 epochs):** Unfreeze `mixed_7a`, `repeat_2`, `block8`, `conv2d_7b` (verified via `model.named_children()`), backbone `lr=1e-5`, heads `lr=1e-3`.
     - **Stage C (optional):** Unfreeze `mixed_6a` and `repeat_1[10:20]` at `lr=5e-6` with layer-wise decay $0.8$ only if validation Macro F1 continues rising.
  5. Override `train()` on the model to force all `BatchNorm2d` layers into `.eval()` mode whenever physical batch size $< 32$; use AdamW (`weight_decay=1e-2`), 1-epoch warmup + cosine schedule, gradient clipping `1.0`, `torch.cuda.amp`, and `channels_last`.
  6. Train across the 5 folds (early stopping patience 3 on validation Macro F1), then train the final candidate on all 12,670 images across 3 random seeds and evaluate once on the test set.
- **Anti-Patterns to Avoid:**
  - Never stack Focal Loss, class weights, and a balanced sampler simultaneously (causes extreme over-prediction of `landslide`/`flood`).
  - Never apply vertical flips to ground-level imagery or heavy Gaussian blur (destroys texture signals).
- **Verification Strategy:**
  - **Explicit Stop Gate:** Must beat Phase 1/2 tuned RF on OOF Macro F1 by $\ge 5.0$ points and on the test set by $>$ 3-seed variance. If not, ship the Phase 1/2 RF model and skip to Phase 5.

---

### Phase 4: Peril-Conditioned Severity Head
- **Goal:** Improve severity classification and DEM consistency by conditioning damage severity on the predicted peril distribution.
- **Risk:** Medium — rare perils (`landslide`, `flood`) have too few severity samples to train standalone heads from scratch.
- **Dependencies:** Phase 3 passing acceptance (every peril recall $\ge 50\%$).
- **Files to Create / Modify:**
  - Modify `models/multitask_net.py` (add `PerilConditionedSeverityHead` with low-rank adapter sharing for rare perils)
- **Implementation Steps:**
  1. Replace the single 3-class severity head with $K=6$ peril-specific branches combined softly:
     $$\mathbf{z}_{\text{severity}} = \sum_{k=1}^6 p(\text{peril}=k) \cdot \text{head}_k(\mathbf{f})$$
  2. Use teacher forcing (ground-truth peril one-hot mixed with predicted $p(\text{peril})$) for the first 3 epochs, annealing to pure predicted probabilities.
  3. Share a low-rank bottleneck adapter across `flood` and `landslide` branches to prevent overfitting.
- **Anti-Patterns to Avoid:**
  - Avoid hard argmax routing (cascades peril classification errors directly into severity).
- **Verification Strategy:**
  - Keep only if OOF severity Macro F1 and DEM correlation improve over the shared severity head.

---

### Phase 5: Calibration, Abstention, TTA, and OOD Guard
- **Goal:** Produce well-calibrated probabilities ($\text{ECE} \le 0.08$), abstain with `'uncertain'` + `top2` when ambiguous, and flag out-of-distribution graphics/memes.
- **Risk:** Low-Medium — TTA increases inference latency on CPU if too many crops are used.
- **Dependencies:** Winning model from Phase 1/2 or Phase 3/4.
- **Files to Create / Modify:**
  - Create `models/calibration.py` (Temperature scaling optimizer, ECE metric, threshold/margin abstention tuner, Mahalanobis OOD detector)
  - Modify `models/hybrid_pipeline.py` (integrate calibration, abstention, optional TTA, and OOD scoring)
- **Implementation Steps:**
  1. Fit a single temperature scalar $T > 0$ per head on OOF logits minimizing negative log-likelihood; record ECE before and after.
  2. Tune per-class probability thresholds $\theta_k$ and top-2 margin threshold $\delta$ on OOF predictions to hit $\ge 90\%$ precision on answered images at $\le 15\%$ abstention rate. When triggered, emit `prediction="uncertain"` and `top2=[class_1, class_2]`.
  3. Benchmark lightweight TTA (original + horizontal flip + 2 crop scales); retain only if OOF Macro F1 improves and CPU latency stays within budget.
  4. Fit class-conditional Gaussian mean and shared covariance on 1,536-D pooled training embeddings to compute Mahalanobis distance (plus max-softmax fallback) as an OOD guard against screenshots/memes.
- **Anti-Patterns to Avoid:**
  - Never fit temperature $T$ or abstention thresholds $\theta_k$ on the test set.
- **Verification Strategy:**
  - Verify OOF and test $\text{ECE} \le 0.08$ and abstention rate $\le 15\%$ at the target precision floor.

---

### Phase 6: Data Expansion for Rare Classes (Parallel Track)
- **Goal:** Expand `flood`, `landslide`, `earthquake`, and `wildfire` datasets so each class has $\ge 300$ labeled test images for tight confidence intervals.
- **Risk:** Medium — web-scraped images can introduce duplicate leakage or license violations.
- **Dependencies:** Runs in parallel with Phases 2–5.
- **Files to Create / Modify:**
  - Modify `step1_collect_data.py` and `step2_preprocess.py` (provenance & license manifest logging, source-disjoint holdout splitting)
  - Create `data/dataset_provenance.csv`
- **Implementation Steps:**
  1. Source additional labeled imagery for `flood`, `landslide`, `earthquake`, and `wildfire` from public open-license disaster datasets and ReliefWeb/GDACS feeds.
  2. Log `image_id, source_url, license, capture_perspective` in `data/dataset_provenance.csv`.
  3. Construct a fresh, **source-disjoint** evaluation holdout set (at least 300 images per rare peril) verified via `pHash` and cosine similarity against all training splits.
- **Anti-Patterns to Avoid:**
  - Never place synthetic or copy-paste augmented images into validation or test splits.
- **Verification Strategy:**
  - Run Phase 0 duplicate detector across the expanded train and source-disjoint test sets to confirm zero leakage.

---

### Phase 7: Production Integration (Flask API, DEM & CrewAI)
- **Goal:** Wire the calibrated, abstention-capable model bundle into `pipeline_singleton.py`, `/api/predict`, Module M2 DEM, and CrewAI without breaking existing UI contracts.
- **Risk:** Medium — DEM scores will shift when fed calibrated severity probabilities instead of uncalibrated RF votes.
- **Dependencies:** Phase 5 calibrated bundle.
- **Files to Create / Modify:**
  - Modify `fusion/feature_fusion.py` (`resolve_artifact_spec` extended for unified calibration/threshold bundles)
  - Modify `crew/pipeline_singleton.py` (add `DISASTERRES_MODEL_BACKEND` toggle for instant rollback behind `ThreadPoolExecutor(max_workers=2)`)
  - Modify `app.py` (extend `/api/predict` with `calibrated_probabilities`, `uncertain`, `top2`, `ood_flag` while preserving legacy fields)
  - Modify `damage_assessment/localization.py` (feed calibrated severity confidence into DEM formula)
  - Modify `crew/report/facts.py`, `crew/report/validator.py`, `crew/tools/aggregator.py`, `crew/crew.py` (propagate `uncertain` and calibrated confidence bands; Analyst requests another image or reports uncertainty when abstaining)
  - Create `tests/test_classification_integration.py`
- **Implementation Steps:**
  1. Package weights, class order, temperatures $T$, thresholds $\boldsymbol{\theta}$, multipliers $\mathbf{w}$, and feature spec into a versioned bundle loaded via `resolve_artifact_spec` (retaining `@lru_cache(maxsize=3)`).
  2. Update `app.py` and `damage_assessment/localization.py` to use calibrated probabilities and expose `uncertain` / `top2`.
  3. Update `crew/crew.py` and `crew/tools/aggregator.py` so confidence bands (`High`, `Medium`, `Moderate`) reflect calibrated probabilities and abstention triggers cautious narrative reporting.
  4. Write golden-image regression tests for all 6 perils, legacy checkpoint loading tests, and CPU latency assertions.
- **Anti-Patterns to Avoid:**
  - Do not remove or rename `prediction`, `confidence`, `probabilities`, or `top5` in `/api/predict` JSON responses (required by `templates/index.html`).
- **Verification Strategy:**
  - Run `pytest tests/test_classification_integration.py` and verify HTTP 200 on `/api/predict` and SSE streaming on `/api/chat`.

---

### Phase 8: Reporting and Release Gates
- **Goal:** Produce the master comparison matrix across all phases and verify every release gate before sign-off.
- **Risk:** Low.
- **Dependencies:** Phase 7.
- **Files to Create / Modify:**
  - Create `reports/phase8_release_comparison.py` and `results/release_comparison_report.md`
  - Modify `README.md` and `docs/project-delivery/learnings.md`
- **Implementation Steps:**
  1. Generate the single comparison table across: Baseline RF, Phase 1 Tuned RF, Phase 2 Best Features, Phase 3 Fine-Tuned, and Phase 5 Calibrated+Abstaining (columns: Macro F1, Balanced Accuracy, Per-Class Recall, Disaster-Miss Rate, ECE, CPU/GPU Latency, Model Size).
  2. Verify all release gates:
     - Every peril recall $\ge 50.0\%$
     - Disaster-to-`not_disaster` miss rate $\le 15.0\%$
     - Calibration $\text{ECE} \le 0.08$
     - CPU/SSE latency within operational budget
  3. Update `README.md` and `learnings.md` to establish Macro F1 and Balanced Accuracy as headline project metrics.
- **Verification Strategy:**
  - Automated gate assertion script exits with code 0 only if all release criteria are met.

---

### Phase 9 (Deferred): SegFormer Damage Segmentation (Enhancement 1)
- **Goal:** Upgrade spatial damage localization from Grad-CAM saliency maps to dense 4-class building segmentation (`no-damage`, `minor`, `major`, `destroyed`) on overhead imagery.
- **Risk:** Medium — license restrictions on xBD (non-commercial) and domain mismatch on ground-level uploads.
- **Dependencies:** Starts **only after Phases 0–8 are complete**.
- **Files to Create / Modify:**
  - Create `crew/vision/perspective_gate.py` (overhead vs. ground-level vs. graphic classifier trained on Phase 0 labels)
  - Create `damage_assessment/segformer_analyzer.py` (SegFormer-B3 / Mask2Former wrapper)
  - Modify `crew/tools/pipeline_tool.py` (route overhead imagery to SegFormer tool; fall back to Grad-CAM Module M2 for ground-level images)
- **Verification Strategy:**
  - Evaluate per-class building IoU and F1 on xBD/FloodNet/Landslide4Sense holdouts and measure agreement with legacy Grad-CAM DEM scores.
