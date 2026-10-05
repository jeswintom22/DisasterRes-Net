# Brainstorm: Fine-Tuning the Classification Model

**Feature Exploration**: Scoping out methods to improve classification performance for the `informativeness` (currently ~78.8% accuracy) and `damage` (currently ~71.5% accuracy) objectives, with the goal of restoring or exceeding the pre-conversion TensorFlow baseline (81% informativeness, 76% damage).

---

## Phase 0: Complexity Assessment
- **Classification**: **Needs Brainstorming**
- **Signals**: Requires modifying model architectures, unfreezing layer parameters, setting up differential optimizer learning rates, adding data augmentation pipelines, tuning dimensionality reduction (PCA), and updating XGBoost training hyperparameters. Touches `step3_classify.py` and experiment tracking.

---

## Phase 1: Context & Current State

### Root Cause of the Accuracy Drop
Upon auditing the git diff between the original TensorFlow version and the new PyTorch version, we discovered the key discrepancy:
- **TensorFlow implementation**: Unfroze the top 50 layers of the backbone:
  ```python
  base.trainable = True
  for layer in base.layers[:-50]:
      layer.trainable = False
  ```
- **PyTorch implementation**: Completely froze the entire backbone, performing only linear probing on the head:
  ```python
  for name, param in model.named_parameters():
      if 'classif' not in name and 'fc' not in name:
          param.requires_grad = False
  ```
This complete freeze restricted the PyTorch backbone from adapting its deep feature representations to the disaster images, causing accuracy to drop from **81% to 78.8% (informativeness)** and **76% to 71.5% (damage)**.

---

## Phase 3: Clarifying Questions & Feedback
1. **Resource/Time Constraints**: Unfreezing and fine-tuning the top layers of the backbone will increase epoch training time from seconds to a few minutes. Since you have an RTX 4060 GPU, 5 epochs of training will take about 1.5–2 minutes per run.
2. **Target Accuracy**: Our immediate goal is to match or exceed the pre-conversion baseline: **81% for informativeness** and **76% for damage**.
3. **Data Augmentation**: We will introduce lightweight training-time augmentations (horizontal flips, slight rotations) to prevent overfitting during fine-tuning.

---

## Phase 4: Proposed Approaches

### Approach 1: Match TensorFlow Setup (Unfreeze last 100 PyTorch Tensors)
Directly unfreeze the last 100 PyTorch parameter tensors (which correspond to the last ~50 Keras layers/blocks in InceptionResNetV2, including `block8` and `conv2d_7b` blocks).

- **How it Works**:
  1. Freeze all backbone parameters by default.
  2. Unfreeze the last 100 parameter tensors (since there are 694 parameter tensors in total in the PyTorch model):
     ```python
     params = list(model.parameters())
     for param in params[-100:]:
         param.requires_grad = True
     ```
  3. Keep the optimizer at `Adam(lr=1e-4)`.
  4. Optionally increase PCA dimensions from 32 to 64 to retain more of the newly learned deep features.
- **Pros**:
  - Parity with the TensorFlow architecture.
  - Likely to immediately recover the lost 3-4% accuracy.
- **Cons**:
  - Epoch training time will increase slightly (from ~10s to ~20s per epoch on RTX 4060).
- **Risk**: Low
- **Effort**: S
- **Likely Files**: `step3_classify.py`

---

### Approach 2: Progressive Fine-Tuning + Differential Learning Rates + Augmentation
Go beyond simple Keras parity by applying modern PyTorch transfer learning best practices to maximize validation accuracy.

- **How it Works**:
  1. Unfreeze the last 100 parameter tensors.
  2. Apply **differential learning rates**: use a lower learning rate for the backbone parameters (`1e-5`) and a higher learning rate for the classifier head (`1e-4`) to prevent gradient shocks.
  3. Add PyTorch data augmentation to `DisasterDataset` training split:
     ```python
     transforms.Compose([
         transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(15),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     ```
  4. Increase PCA dimensions to `128` to fuse richer deep features with the GLCM features.
- **Pros**:
  - Prevents overfitting and yields better generalization than Keras baseline.
  - Differential learning rates preserve pretrained features while adapting top layers safely.
- **Cons**:
  - Requires implementing separate training/validation transform pipelines.
- **Risk**: Low-Medium
- **Effort**: M
- **Likely Files**: `step3_classify.py`

---

### Approach 3: Hyperparameter Optimization for XGBoost (No Backbone Unfreezing)
Keep the backbone completely frozen (zero extra training cost) and instead try to close the accuracy gap by tuning XGBoost and feature selection.

- **How it Works**:
  1. Perform feature extraction on the frozen backbone.
  2. Use `SelectKBest(f_classif, k=64)` instead of PCA to select the best deep features.
  3. Tune XGBoost using a grid search over `learning_rate` (0.01 to 0.1), `max_depth` (4 to 8), and `subsample` (0.8 to 1.0).
- **Pros**:
  - Extremely fast execution; no risk of training divergence.
- **Cons**:
  - Less likely to close the gap because the underlying feature representation cannot adapt to disaster categories.
- **Risk**: Low
- **Effort**: S
- **Likely Files**: `step3_classify.py`

---

## Phase 5: Recommended Delivery Path
I recommend **Approach 1 (Match TensorFlow Setup)** first as a baseline validation to confirm we restore the 76% (damage) and 81% (informativeness) accuracy. Once verified, we can apply **Approach 2**'s augmentations and learning rate scheduling to push performance even higher.
