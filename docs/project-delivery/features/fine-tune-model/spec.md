# Functional Specification: Fine-Tuning the Classification Model

**Status**: Draft
**Feature Area**: step3_classify.py (Model Training and Evaluation Pipeline)

---

## Overview
This specification defines the implementation of deep fine-tuning for the PyTorch InceptionResNetV2 image classification model. By unfreezing the top layers, applying differential learning rates, implementing data augmentation, and increasing PCA components, we aim to recover and exceed the pre-conversion TensorFlow baseline performance (81% accuracy for informativeness, 76% accuracy for damage).

---

## What Exists Today
- A fully migrated PyTorch pipeline in `step3_classify.py` using `timm.create_model('inception_resnet_v2', pretrained=True)`.
- The backbone is **100% frozen** (only the classifier head is trained).
- Deep features are extracted, reduced to 32 dimensions via PCA, combined with 20 GLCM texture features, and trained using XGBoost.
- Accuracy is currently at **78.8% for informativeness** and **71.5% for damage** (a drop from the TensorFlow baseline).
- Data loading uses standard ImageNet resizing and normalization without training-time data augmentations.

---

## What Changes

### 1. Data Transform Pipeline
- Training transforms will include **data augmentation** to mitigate overfitting during backbone fine-tuning.
- Testing/validation transforms will remain deterministic (resizing and normalization only).

### 2. Selective Backbone Parameter Unfreezing
- Instead of freezing all backbone parameters, the last **100 PyTorch parameter tensors** (corresponding to the top ~50 Keras layers/blocks, including `block8` and `conv2d_7b` blocks) will be unfrozen.

### 3. Differential Learning Rates
- We will group the learnable parameters in the optimizer:
  - **Backbone parameters**: Low learning rate (`1e-5`) to adapt high-level features gently.
  - **Classifier head parameters**: Standard learning rate (`1e-4`) to learn the new classes quickly.

### 4. PCA Dimensionality Increase
- Increase PCA dimensions from `32` to `64` to preserve more of the rich representations learned during deep fine-tuning.

---

## Functional Requirements

- **FR-1 (Augmented Training Transforms)**: 
  - `get_transforms(train: bool)` must return:
    - **Train**: `transforms.Resize((299, 299))`, `transforms.RandomHorizontalFlip()`, `transforms.RandomRotation(15)`, `transforms.ToTensor()`, `transforms.Normalize(...)`.
    - **Eval**: `transforms.Resize((299, 299))`, `transforms.ToTensor()`, `transforms.Normalize(...)`.
- **FR-2 (Backbone Parameter Selection)**:
  - The model parameters will be frozen by default (`param.requires_grad = False`).
  - The last 100 parameter tensors of the instantiated `timm` model list must be set to `param.requires_grad = True`.
- **FR-3 (Differential Optimizer Configuration)**:
  - In `finetune_extractor`, learnable parameters (`param.requires_grad == True`) must be partitioned:
    - If the parameter name contains `classif` or `fc`, it belongs to the classifier head (`lr=1e-4`).
    - Otherwise, it belongs to the backbone (`lr=1e-5`).
  - The optimizer must be instantiated with these parameter groups.
- **FR-4 (Dimensionality Adaptation)**:
  - Modify `step3_classify.py` to use `n_components=64` in `PCA`.
  - Update PCA feature column naming (`deep_pca_1` to `deep_pca_64`) and related logging fields.

---

## Edge Cases
- **Stochastic Eval Leakage**: Ensuring evaluation feature extraction uses `train=False` transforms so that test predictions remain deterministic and reproducible.
- **OOM during Training**: Unfreezing parameters allocates more memory on the GPU for gradients. We must ensure batch size (64) is stable; if OOM occurs, handle gradient zeroing and empty cache cleanly.

---

## Out of Scope
- Changing the backbone architecture from `inception_resnet_v2` to another network.
- Hyperparameter search for XGBoost parameters (XGBoost settings will remain as configured).

---

## Dependencies
- PyTorch, Torchvision, TIMM, Scikit-Learn (already installed in the `disaster_Net` virtual environment).

---

## Acceptance Criteria

- **AC-1 (Pipeline Run)**: `step3_classify.py` runs successfully to completion for both objectives (`informativeness` and `damage`) and logs all artifacts and metrics to MLflow.
- **AC-2 (Informativeness Accuracy)**: The final test accuracy for the informativeness objective must meet or exceed **81.0%**.
- **AC-3 (Damage Accuracy)**: The final test accuracy for the damage objective must meet or exceed **76.0%**.
- **AC-4 (GPU Utilization)**: The model fine-tuning process must run on the GPU when CUDA is available.
