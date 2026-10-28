# Functional Specification: DisasterRes-Net Classification Improvement (Peril, Severity & Calibration)

**Feature ID:** `003-classification-improvement`  
**Status:** Approved (Baseline Spec from Parallax Machine Inc. PDF)  
**Target Modules:** `step3_classify.py`, `fusion/feature_fusion.py`, `models/hybrid_pipeline.py`, `damage_assessment/localization.py`, `app.py`, `crew/pipeline_singleton.py`, `crew/crew.py`, `crew/report/facts.py`, `crew/tools/aggregator.py`

---

## 1. Overview

The current DisasterRes-Net peril classifier (`disaster_type` head using a frozen `timm` Inception-ResNet-V2 backbone, 2,036-D early-fused features, and a 300-tree Random Forest) achieves **68.45% overall accuracy** on the 5,433-image test set (and **68.06% ± 1.06%** on 5-fold stratified cross-validation across 12,670 training images). However, it suffers from severe decision-rule collapse on minority perils:
- **Macro Recall is 35.69%** and **Macro F1 is 40.21%**.
- **96.3% of all predictions** fall into just two classes: `hurricane` (2,201) or `not_disaster` (3,031).
- **Rare perils have high precision but near-zero recall:** `flood` (100% precision, 2.08% recall), `landslide` (100% precision, 7.14% recall), `earthquake` (90.59% precision, 19.90% recall), `wildfire` (91.96% precision, 33.33% recall).
- **Critical Emergency Failure Mode:** 908 out of 2,880 real disaster test images (**31.5%**) are misclassified as `not_disaster`.
- **Dominant Confusion Pair:** Confusion between `hurricane` and `not_disaster` accounts for 1,062 of the 1,714 total test errors (**62.0%**).

This specification defines a disciplined, 10-phase (Phases 0–9) improvement program that diagnoses data/label noise, repairs the Random Forest decision rule via out-of-fold (OOF) post-processing, fixes `timm` Inception-ResNet-V2 input normalization and feature pooling, conducts an end-to-end multi-task fine-tuning bake-off, adds peril-conditioned severity, calibrates probabilities with an explicit `'uncertain'` abstention path, and integrates cleanly into the Flask API and CrewAI reporting pipeline.

---

## 2. What Exists Today

1. **Feature Extraction & Normalization (`step3_classify.py`):**
   - Uses `timm.create_model("inception_resnet_v2", pretrained=True)` with default `num_classes=1000`, outputting 1,000-D ImageNet **classifier logits** rather than 1,536-D penultimate pooled embeddings.
   - Normalizes inputs with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`, whereas `timm`'s pretrained `inception_resnet_v2` weights expect `mean=[0.5, 0.5, 0.5]` and `std=[0.5, 0.5, 0.5]`.
   - Concatenates `df1` (1000-D original CNN logits), `df2` (1000-D LMS saliency CNN logits), `glcm` (20-D), and `lbp_stats` (16-D) into a 2,036-D vector (`df1_df2_glcm_lbp_stats`).
2. **Training & Cross-Validation (`step3_classify.py`):**
   - Fits `StandardScaler` on the full `x_train` matrix *before* running `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, causing mild in-fold mean/variance leakage during CV.
   - Trains `RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42)` with unpruned trees (`min_samples_leaf=1`), which memorizes training leaves and outputs near-zero probabilities for rare classes on unseen data.
   - Evaluates `cross_val_score(..., scoring="accuracy")` only, ignoring `f1_macro` and `balanced_accuracy`.
3. **Inference & Post-Processing (`models/hybrid_pipeline.py`, `app.py`, `crew/`):**
   - Predicts via standard `argmax` over raw uncalibrated Random Forest probabilities (`clf.predict(fused_scaled)`).
   - No abstention mechanism (`'uncertain'`) or out-of-distribution (OOD) guard exists.
   - `DamageLocalizationAnalyzer` (Module M2) computes DEM using raw uncalibrated severity confidence.
   - `crew/tools/aggregator.py` derives confidence bands (`high`, `medium`, `moderate`) from raw damage confidence without temperature calibration.

---

## 3. What Changes

1. **Rigorous Evaluation Protocol (All Phases):**
   - Touch the 5,433-image test set **at most once per phase candidate** after all hyperparameter and threshold tuning is locked.
   - Perform all tuning on the 12,670 training images using out-of-fold predictions (`cross_val_predict`) over `StratifiedKFold(5, shuffle=True, random_state=42)` with `StandardScaler` encapsulated inside an `sklearn.pipeline.Pipeline`.
   - Replace `accuracy` with **Macro F1**, **Balanced Accuracy**, **Per-Class Recall**, and **Disaster-to-Not-Disaster Miss Rate** as primary metrics, reporting fold spreads and Wilson/bootstrap 95% confidence intervals for classes with $<100$ samples.
2. **Phase 0 (Diagnostics & Data Audit):**
   - Quantify the train-vs-OOF memorization gap, class imbalance ratios (73.6:1 `hurricane` to `landslide`), cross-split duplicate leakage via perceptual hashes (`pHash`/`dHash`) and cosine similarity, label noise across the top 4 confusion buckets (200-image audit), and perspective distribution (overhead vs. ground-level vs. graphic).
3. **Phase 1 (Random Forest Decision-Rule & Regularization Fix):**
   - Regularize RF leaf depth (`class_weight='balanced_subsample'`, grid over `min_samples_leaf` $\in \{1, 3, 5, 10\}$ and `max_features` $\in \{\text{'sqrt'}, 0.1\}$).
   - Learn per-class probability multipliers $\mathbf{w} \in [0.5, 20]^6$ (or prior-correction exponent $\tau$) via coordinate search on OOF probabilities to maximize OOF Macro F1, plus a cost-sensitive variant penalizing disaster-to-`not_disaster` errors ($2\times$–$3\times$).
4. **Phase 2 (Feature Upgrade & Stream Ablation):**
   - Align input transforms with `timm.data.resolve_data_config` (`mean=[0.5]*3, std=[0.5]*3`).
   - Extract 1,536-D pooled features (`num_classes=0`) and optional `mixed_6a` 1,088-D intermediate texture features; register a versioned feature spec (`v3`) in `resolve_artifact_spec` while preserving backward compatibility for 2,030-D and 2,036-D checkpoints.
   - Run a systematic OOF ablation across streams (`df1`, `df2`, `glcm`, `lbp_stats`) and drop any stream (including LMS saliency) that fails to improve OOF Macro F1 by $>1.0$ point.
5. **Phase 3 (End-to-End Multi-Task Fine-Tuning):**
   - Train a shared Inception-ResNet-V2 backbone with three multi-task heads (`disaster_type` [6], `damage` [3], `informativeness` [2]) using progressive 3-stage unfreezing (Stage A: heads only; Stage B: `mixed_7a`, `repeat_2`, `block8`, `conv2d_7b`; Stage C: `mixed_6a` + upper `repeat_1`), comparing three non-stacked imbalance arms (Focal + Effective Number vs. Logit-Adjusted CE vs. Class-Balanced Sampler + CE), with BatchNorm statistics frozen in `eval()` mode when batch size $<32$.
6. **Phase 4 (Peril-Conditioned Severity Head):**
   - Replace the single severity head with $K$ peril-conditioned severity heads combined softly via $\sum_k p(\text{peril}=k) \cdot \text{head}_k(\mathbf{z})$, sharing a low-rank adapter/fallback for rare perils (`landslide`, `flood`).
7. **Phase 5 (Calibration, Abstention, TTA & OOD Guard):**
   - Fit per-head temperature scaling $T$ on OOF logits, enforce threshold/margin-based abstention (`'uncertain'` + `top2` classes), evaluate test-time augmentation (horizontal flip + multi-scale crops), and attach a Mahalanobis / max-softmax OOD detector.
8. **Phase 6–9 (Data Expansion, Production Integration, Release Gates & Deferred SegFormer):**
   - Expand minority classes (`flood`, `landslide`, `earthquake`, `wildfire`) to $\ge 300$ test samples in a source-disjoint holdout set; integrate calibrated bundles into `pipeline_singleton.py`, `/api/predict`, DEM, and CrewAI `FactsBlock`; enforce release gates; defer SegFormer building segmentation to Phase 9 behind the perspective gate.

---

## 4. Functional Requirements

### Evaluation & Governance
- **FR-1 (Zero Test Leakage & In-Fold Scaling):** All scalers, feature selectors, probability multipliers, and calibration temperatures MUST be fitted strictly inside each cross-validation fold (`sklearn.pipeline.Pipeline` or explicit fold indexing) on the 12,670 training samples. The 5,433-image test set MUST only be evaluated once per locked phase winner.
- **FR-2 (Expanded MLflow Telemetry):** Every training and evaluation run MUST log `cv_f1_macro_mean`, `cv_f1_macro_std`, `cv_balanced_acc_mean`, `cv_balanced_acc_std`, per-class precision/recall/F1 per fold, 95% confidence intervals for classes with $<100$ support, disaster-to-`not_disaster` miss rate, git commit hash, and feature spec version.

### Phase 0: Diagnostics & Data Audit
- **FR-3 (Memorization & Imbalance Audit):** Provide an automated diagnostic script (`evaluation/phase0_diagnostics.py`) that outputs train vs. OOF vs. test scores, exact class counts, and imbalance ratios across peril and severity splits.
- **FR-4 (Cross-Split Duplicate & Leakage Detection):** Compute perceptual hashes (`pHash`/`dHash`, Hamming distance $\le 6$) and L2-normalized CNN embedding cosine similarity ($\ge 0.98$) across all 18,103 train+test images. Report cross-split near-duplicate pairs and provide a `GroupKFold` cluster assignment.
- **FR-5 (Stratified Error & Perspective Audit):** Export a 200-image audit manifest (50 samples each from `hurricane` $\rightarrow$ `not_disaster`, `not_disaster` $\rightarrow$ `hurricane`, `earthquake` $\rightarrow$ `hurricane`, `wildfire` $\rightarrow$ `hurricane`) for tagging (`model-wrong`, `label-wrong`, `ambiguous`) and perspective (`overhead`, `ground-level`, `graphic`). If `label-wrong` exceeds 20%, trigger automated/semi-automated label cleaning before Phase 3.

### Phase 1: Random Forest Decision-Rule & Regularization
- **FR-6 (RF Regularization Grid):** Evaluate `RandomForestClassifier` with `class_weight="balanced_subsample"`, `min_samples_leaf` $\in \{1, 3, 5, 10\}$, `max_features` $\in \{\text{"sqrt"}, 0.1\}$, and `n_estimators` $\in \{300, 500\}$ using 5-fold `cross_val_predict(method="predict_proba")` inside an in-fold `StandardScaler` pipeline.
- **FR-7 (OOF Probability Multipliers & Cost-Sensitive Tuning):** Implement coordinate ascent and prior-adjustment ($P_k / f_k^\tau$) on OOF probability matrices to learn per-class weights $\mathbf{w} \in [0.5, 20]^6$ maximizing OOF Macro F1, alongside a cost-sensitive matrix that penalizes predicting `not_disaster` on true disasters by factor $\lambda \in [2.0, 3.0]$. Persist $\mathbf{w}$ inside the artifact bundle.

### Phase 2: Feature Upgrade & Stream Ablation
- **FR-8 (TIMM Native Normalization & 1536-D Pooled Extraction):** Construct preprocessing transforms via `timm.data.resolve_data_config` (`299x299`, `mean=(0.5, 0.5, 0.5)`, `std=(0.5, 0.5, 0.5)`) and instantiate `inception_resnet_v2` with `num_classes=0` to extract 1,536-D pooled representations (plus optional 1,088-D globally pooled `mixed_6a` features).
- **FR-9 (Versioned Feature Contract & Stream Ablation):** Register feature spec `v3` in `fusion/feature_fusion.py` (`resolve_artifact_spec`) without breaking legacy 2,030-D (`df1_df2_glcm_lbp`) or 2,036-D (`df1_df2_glcm_lbp_stats`) checkpoints. Run the full OOF ablation matrix (`df1`, `df1+df2`, `df1+glcm+lbp`, all four; 1000-D vs. 1536-D vs. 1536-D+1088-D) and select the most compact feature spec within 1.0 Macro F1 point of the maximum.

### Phase 3: End-to-End Multi-Task Fine-Tuning
- **FR-10 (Shared Multi-Task Architecture):** Implement a PyTorch `MultiTaskDisasterNet` wrapping `inception_resnet_v2` (`num_classes=0`) with three classification heads (`disaster_type`: 6, `damage`: 3, `informativeness`: 2), configurable dropout ($0.3$–$0.5$), label smoothing ($0.05$–$0.10$), and optional concatenation of surviving handcrafted features before the linear layers.
- **FR-11 (Mutually Exclusive Imbalance Arms & Masked Loss):** Support three selectable imbalance training arms—(a) Focal Loss ($\gamma=2$) with effective-number $\alpha$ ($\beta=0.999$), (b) Logit-Adjusted Cross-Entropy ($z_k + \tau \log \pi_k$), and (c) Class-Balanced Sampler with standard Cross-Entropy—combined via $L = L_{\text{peril}} + 0.5 L_{\text{damage}} + 0.25 L_{\text{info}}$ with missing-label masking.
- **FR-12 (3-Stage Progressive Unfreezing & BN Freeze):** Implement Stage A (3–5 epochs, frozen backbone, head LR `1e-3`), Stage B (5–10 epochs, unfreeze `mixed_7a`, `repeat_2`, `block8`, `conv2d_7b` verified via `named_children()`, backbone LR `1e-5`), and optional Stage C (`mixed_6a` + `repeat_1[10:20]`, backbone LR `5e-6`, layer decay `0.8`), using AdamW (`weight_decay=1e-2`), cosine annealing with 1-epoch warmup, grad clipping `1.0`, AMP, and forced `eval()` mode on all `BatchNorm2d` layers when batch size $< 32$.

### Phase 4 & 5: Peril-Conditioned Severity, Calibration, Abstention & OOD
- **FR-13 (Soft Peril-Conditioned Severity Head):** Implement $K=6$ peril-conditioned severity branches combined via $\mathbf{z}_{\text{sev}} = \sum_{k=1}^6 p(\text{peril}=k) \cdot \text{head}_k(\mathbf{f})$, using teacher-forcing mixture schedules during initial epochs and a shared low-rank adapter for `flood` and `landslide`.
- **FR-14 (Temperature Scaling, Abstention & OOD Guard):** Fit scalar temperature $T > 0$ per head via NLL minimization on OOF logits (targeting $\text{ECE} \le 0.08$). Emit `prediction="uncertain"` with `top2` candidate classes whenever $\max_k \tilde{p}_k < \theta_k$ or $(\tilde{p}_{(1)} - \tilde{p}_{(2)}) < \delta$, and compute Mahalanobis / max-softmax OOD scores to flag non-disaster graphics or screenshots.

### Phase 6, 7 & 8: Data Expansion, API/CrewAI Integration & Release Gates
- **FR-15 (Backward-Compatible Artifact Bundle & Rollback Toggle):** Store model weights, class order, temperature scalars $T$, per-class thresholds $\boldsymbol{\theta}$, probability multipliers $\mathbf{w}$, and feature spec metadata in a unified bundle. Add `DISASTERRES_MODEL_BACKEND=legacy_rf|calibrated_rf|multitask_nn` in `crew/pipeline_singleton.py` behind the existing `ThreadPoolExecutor(max_workers=2)`.
- **FR-16 (API, DEM & CrewAI Uncertainty Propagation):**
  - Extend `POST /api/predict` response with `calibrated_probabilities`, `uncertain` (boolean), `top2` (list), and `ood_flag` while keeping all existing JSON keys intact for `templates/index.html`.
  - Feed calibrated severity confidence into `DamageLocalizationAnalyzer` (Module M2 DEM calculation) and document the DEM baseline shift.
  - Extend `FactsBlock` and `NumericFidelityValidator` with `uncertain` and `calibrated_confidence`; when `uncertain=True`, instruct the CrewAI Analyst/Scout pipeline to request additional imagery or report uncertainty rather than asserting a single peril.

---

## 5. Edge Cases

1. **Zero-Prediction Fold for Rare Classes (`landslide`, `flood`):** During OOF coordinate search for multipliers $\mathbf{w}$, unconstrained optimization can overfit the 28-sample `landslide` class. Multipliers MUST be box-constrained to $[0.5, 20.0]$ and regularized via nested OOF or smoothed prior powers $\tau \in [0.2, 1.5]$.
2. **BatchNorm Instability under Gradient Accumulation:** Fine-tuning `inception_resnet_v2` with physical batch sizes of 8 or 16 on consumer GPUs corrupts running mean/variance in `BatchNorm2d` layers. Overriding `model.train()` to keep all `nn.modules.batchnorm._BatchNorm` modules in `.eval()` mode is mandatory when physical batch size $< 32$.
3. **Windows Multiprocessing & OpenBLAS Contention:** As documented in `learnings.md`, spawning workers on Windows with top-level CUDA initialization or unconstrained OpenBLAS threads causes WDDM resets or memory allocation crashes. CUDA initialization must remain lazy/local, and `OPENBLAS_NUM_THREADS=1` must be enforced.
4. **Legacy Checkpoint Fallback:** If a user or test loads legacy `rf_damage_df1_df2_glcm_lbp.joblib` (2,030-D) or `rf_disaster_type_df1_df2_glcm_lbp_stats.joblib` (2,036-D) without calibration parameters, the loader must default to $\mathbf{w}=\mathbf{1}, T=1.0, \theta_k=0.0$ seamlessly.
5. **High-Confidence Non-Aerial Graphics (Memes/Screenshots):** News scraping (`GoogleImageScrapeTool`) frequently returns news banners or maps that pass basic resolution checks. The Phase 5 Mahalanobis/OOD guard and Phase 0/9 perspective gate must flag these before they distort aggregated DEM scores.

---

## 6. Out of Scope

- Replacing the CrewAI orchestration architecture or SSE streaming transport.
- Synthetic Aperture Radar (Sentinel-1 SAR) or multispectral (Sentinel-2 NIR/SWIR) band fusion.
- Bi-temporal pre/post-disaster Siamese change-detection networks.
- Switching the primary CNN backbone away from `inception_resnet_v2` (DINOv2/CLIP linear probes are restricted to optional Phase 2 reference baselines only if `inception_resnet_v2` plateaus).
- Building-level SegFormer segmentation during Phases 0–8 (explicitly deferred to Phase 9 after classification release gates pass).

---

## 7. Dependencies

- **Existing Packages:** `torch`, `torchvision`, `timm`, `scikit-learn`, `scikit-image`, `numpy`, `scipy`, `joblib`, `mlflow`, `Pillow`, `opencv-python`, `flask`.
- **Optional / Lightweight Additions:** `imagehash` (already used in `crew/vision/` for perceptual hashing in Phase 0).
- **Data Assets:** `split_dataset/train` (12,670 images), `split_dataset/test` (5,433 images), cached `.npz` feature matrices in `mlruns/2/`.

---

## 8. Acceptance Criteria

- **AC-1 (Phase 0 Audit Completion):** Diagnostic report generated with train vs. OOF accuracy gap, cross-split duplicate count, 200-image label noise percentage across the 4 primary confusion buckets, and perspective breakdown.
- **AC-2 (Phase 1 Decision-Rule Gate):** On 5-fold OOF and single-touch test evaluation, Macro F1 increases by **$\ge 10$ percentage points** over the 40.21% baseline (reaching $\ge 50.2\%$, target $\ge 55.0\%$), `flood` and `landslide` recall reach **$\ge 30.0\%$**, and no peril's recall drops by $> 5.0$ points from baseline.
- **AC-3 (Phase 2 Feature & Normalization Gate):** Input preprocessing matches `timm.data.resolve_data_config` (`mean=0.5, std=0.5`), 1,536-D pooled features are benchmarked in MLflow against 1,000-D logits, and the chosen feature spec `v3` is within $1.0$ Macro F1 point of the richest combination while legacy 2,030-D/2,036-D models still load in unit tests.
- **AC-4 (Phase 3 Fine-Tuning Gate):** End-to-end fine-tuned model outperforms Phase 1/2 tuned RF on OOF Macro F1 by **$\ge 5.0$ points** and on the test set by more than 3-seed variance (targeting Macro F1 $\ge 70.0\%$, every class recall $\ge 50.0\%$, and disaster-to-`not_disaster` miss rate $\le 15.0\%$). If it does not beat Phase 1/2 RF by this margin, the system ships the Phase 1/2 model.
- **AC-5 (Phase 5 Calibration & Abstention Gate):** Post-temperature scaling Expected Calibration Error ($\text{ECE}$) is **$\le 0.08$**, and the abstention threshold achieves the target precision/abstention operating point (e.g., $\ge 90\%$ precision on non-abstained predictions at $\le 15\%$ abstention rate).
- **AC-6 (Phase 7 & 8 Production Release Gate):** `POST /api/predict` and CrewAI chat flows return calibrated probabilities, `uncertain` flags, and `top2` perils without breaking `templates/index.html` or `frontend/src/App.jsx`, passing golden-image regression tests and CPU latency budgets.

---

## 9. Glossary

- **OOF (Out-of-Fold):** Predictions generated for each training fold using a model trained exclusively on the remaining $K-1$ folds (`cross_val_predict`).
- **Macro F1:** Unweighted arithmetic mean of per-class F1 scores, treating rare perils (`landslide`, `flood`) equally with dominant classes (`hurricane`, `not_disaster`).
- **Disaster-to-Not-Disaster Miss Rate:** Fraction of true disaster images (`earthquake`, `flood`, `hurricane`, `landslide`, `wildfire`) falsely predicted as `not_disaster` (baseline: $908 / 2880 = 31.53\%$).
- **ECE (Expected Calibration Error):** Weighted average gap between predicted confidence and empirical accuracy across probability bins.
- **DEM (Damage Evaluation Metric):** Composite 0–100 spatial severity score combining affected area percentage, damage density, texture entropy, and classifier severity confidence.

