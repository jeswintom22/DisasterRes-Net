# Architectural & Technical Decisions: Classification Improvement

**Feature ID:** `003-classification-improvement`  
**Date:** 2026-10-28

---

## ADR-001: Prioritize Decision-Rule Repair (Phase 1) Before Backbone Retraining (Phase 3)
- **Context:** On the 5,433-image test set, minority perils exhibit high precision (`flood` 100%, `landslide` 100%, `wildfire` 92.0%, `earthquake` 90.6%) alongside very low recall (2.1%–33.3%). Furthermore, 96.3% of predictions are `hurricane` or `not_disaster`.
- **Decision:** Execute Phase 1 (`balanced_subsample`, leaf regularization `min_samples_leaf` $\in [3, 5]$, and OOF probability multiplier tuning $\mathbf{w}$) on the existing cached feature matrices *before* attempting GPU-intensive backbone fine-tuning.
- **Consequences:** High precision proves the 2,036-D features already separate minority perils in feature space, but unpruned Random Forest trees (`min_samples_leaf=1`) and uniform argmax thresholds suppress minority posterior probabilities. Fixing the decision rule yields an immediate Macro F1 jump on CPU at near-zero compute cost.

## ADR-002: Strict In-Fold Scaling and Single-Touch Test Set Governance
- **Context:** Previously, `step3_classify.py` fitted `StandardScaler` on the full `x_train` array before running `cross_val_score`, and evaluated models directly on `x_test`.
- **Decision:** All preprocessing (`StandardScaler`), threshold search ($\mathbf{w}$), hyperparameter selection, and temperature scaling ($T$) MUST run on 5-fold out-of-fold (OOF) predictions (`StratifiedKFold(5, shuffle=True, random_state=42)`) inside an `sklearn.pipeline.Pipeline`. The 5,433-image test set is touched strictly once per locked phase winner.
- **Consequences:** Eliminates optimistic variance leakage and prevents threshold overfitting on the 28-sample `landslide` test class.

## ADR-003: Replace 1,000-D ImageNet Logits with 1,536-D Pooled Features & Native TIMM Normalization
- **Context:** `step3_classify.py` currently normalizes images with ImageNet mean/std (`[0.485, 0.456, 0.406]`) and extracts the 1,000-D output of `inception_resnet_v2`'s ImageNet classification head. However, `timm`'s `inception_resnet_v2` pretrained weights expect `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]` and provide a 1,536-D penultimate pooled representation when instantiated with `num_classes=0`.
- **Decision:** Build transforms dynamically from `timm.data.resolve_data_config` and extract 1,536-D pooled features (plus optional 1,088-D `mixed_6a` pooled texture features) under a new versioned feature contract (`v3`).
- **Consequences:** Eliminates input distribution shift and avoids compressing crisis visual features through 1,000 unrelated ImageNet object categories. Legacy 2,030-D and 2,036-D artifacts remain loadable via `resolve_artifact_spec`.

## ADR-004: Mutually Exclusive Imbalance Arms & Frozen BatchNorm in Phase 3
- **Context:** Stacking Focal Loss, class weights, and class-balanced sampling simultaneously causes severe over-correction and collapses precision on `hurricane` and `not_disaster`. Additionally, fine-tuning a 56M-parameter network with small physical batch sizes ($<32$) degrades `BatchNorm2d` running statistics.
- **Decision:** Benchmark three mutually exclusive imbalance arms—(a) Focal Loss ($\gamma=2$) + effective-number $\alpha$, (b) Logit-Adjusted CE, and (c) Class-Balanced Sampler + plain CE—while forcing all `BatchNorm2d` modules into `.eval()` mode during training whenever physical batch size $<32$. Include an explicit stop gate: if Phase 3 does not beat Phase 1/2 OOF Macro F1 by $\ge 5.0$ points, ship the Phase 1/2 model.
- **Consequences:** Prevents training divergence and protects production deployment from shipping an overfitted deep model if the regularized Random Forest performs comparably on this dataset scale.

## ADR-005: Soft Peril-Conditioned Severity Routing Over Hard Router-Plus-Specialist
- **Context:** Damage appearance depends heavily on the underlying peril (e.g., wildfire burn scars vs. earthquake rubble), but hard routing cascades peril classification errors into severity scoring and starves rare perils (`landslide`, `flood`) of training data.
- **Decision:** Combine $K$ peril-specific severity heads softly via $\sum_k p(\text{peril}=k) \cdot \text{head}_k(\mathbf{f})$, sharing a low-rank adapter for `flood` and `landslide`.
- **Consequences:** Keeps training data pooled across the shared representation and avoids hard routing errors.

## ADR-006: Calibrated Abstention (`'uncertain'`) and Deferral of SegFormer to Phase 9
- **Context:** Asserting a wrong peril or predicting `not_disaster` on an active disaster is dangerous for emergency responders. Meanwhile, SegFormer building segmentation only applies to overhead imagery and does not solve peril classification.
- **Decision:** Add temperature scaling ($T$) and an explicit `'uncertain'` abstention state (returning `top2` classes when confidence or margin is low), propagating uncertainty into `FactsBlock` so the CrewAI Analyst requests additional evidence. Defer SegFormer B3 to Phase 9 behind an overhead perspective gate.
- **Consequences:** Ensures high precision ($\ge 90\%$) on answered predictions while keeping the core peril/severity upgrade focused and unblocked.
