# Status: Fine-Tuning classification model

## Phases

- [x] **Phase 1: Update Data Transforms & Deep Extraction**
  - Implement augmented transforms in `get_transforms` with a `train` flag.
  - Implement deterministic evaluation transforms for validation feature extraction.
- [x] **Phase 2: selective Backbone Parameter Unfreezing & Differential Learning Rates**
  - Freeze all backbone parameters by default.
  - Unfreeze the last 100 parameter tensors.
  - Group learnable parameters into backbone (`lr=1e-5`) and classifier head (`lr=1e-4`).
- [x] **Phase 3: Model and Endpoint Compatibility Checkpoints**
  - Save full PyTorch `model_{objective}.pth` before resetting the classifier.
  - Save duplicate checkpoints (`model_disaster_type.pth` and `le_disaster_type.joblib`) for web application compatibility.
- [x] **Phase 4: PCA Expansion to 64 Components**
  - Increase components from 32 to 64.
  - Update feature naming and log parameters/metrics.
- [x] **Phase 5: demo.py Framework Porting**
  - Migrate `demo.py` to PyTorch/XGBoost.
  - Test prediction on a sample image.
