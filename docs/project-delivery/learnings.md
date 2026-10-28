# Project Learnings

## PyTorch Multiprocessing Crashes on Windows (spawn)
- **Problem**: On Windows, the default multiprocessing start method is `spawn`. This causes all child processes to re-import the main module. If PyTorch CUDA (`device = torch.device('cuda')`) is initialized at the top-level (module scope), every worker process spawned (e.g. for GLCM extraction) will attempt to create a CUDA context simultaneously. This exhausts GPU resources and triggers WDDM GPU driver resets (TDR, leading to black screens) and hard memory access violations (`python.exe - Application Error`).
- **Solution**: Always initialize the CUDA device lazily or within local functions where training/inference takes place. Protect CUDA code from running in spawned child processes.

## OpenBLAS Memory Allocation Failures in Multiprocessing
- **Problem**: Large process pools (e.g. 32 workers) executing image extraction operations (like GLCM via `scikit-image`) can trigger OpenBLAS out-of-memory errors (`OpenBLAS error: Memory allocation still failed after 10 retries, giving up.`). This is caused by multiple workers attempting to allocate memory for internal thread pools concurrently while decoding and transforming images.
- **Solution**: 
  1. Limit the maximum number of workers to a conservative value (e.g., `min(8, os.cpu_count())`).
  2. Disable OpenBLAS multithreading for child processes by setting the environment variables:
     ```python
     os.environ["OPENBLAS_NUM_THREADS"] = "1"
     os.environ["OMP_NUM_THREADS"] = "1"
     os.environ["MKL_NUM_THREADS"] = "1"
     os.environ["NUMEXPR_NUM_THREADS"] = "1"
     ```
     This forces sequential execution within each worker process, preventing the ballooning of thread pool allocations.


## Severe Class Imbalance & Random Forest Vote-Share Calibration (Feature 003)
- **Problem**: Under a `93:1` class imbalance (`not_disaster` = 5,957 vs. `landslide` = 64 in training), an unregularized `RandomForestClassifier(min_samples_leaf=1, class_weight='balanced')` memorizes 100.00% of the training set (100% Train Macro F1 vs. 40.21% Test Macro F1). At inference time, `predict()` averages raw leaf probabilities across 300 trees where majority classes (`not_disaster`, `hurricane`) dominate vote shares, causing `97.9%` of `flood` and `92.9%` of `landslide` test images to be missed.
- **Solution**:
  1. **In-Fold Regularization**: Wrap `StandardScaler()` and `RandomForestClassifier(class_weight="balanced_subsample", min_samples_leaf=3, max_features="sqrt")` inside an `sklearn.pipeline.Pipeline` during 5-fold `StratifiedKFold` cross-validation so scaler statistics never leak across folds.
  2. **OOF Probability Multipliers & Temperature Scaling**: Fit per-class post-hoc multipliers $\mathbf{w} \in [0.5, 20.0]^6$ and scalar temperature $T > 0$ via coordinate ascent on 5-fold out-of-fold (OOF) probability predictions, plus a Mahalanobis OOD guard and top-2 margin abstention (`'uncertain'`), stored in `CalibratedModelBundle`.

## Inception-ResNet-v2 Native Normalization & Penultimate Representation (`timm`)
- **Problem**: `timm`'s `inception_resnet_v2` weights were trained with `mean=(0.5, 0.5, 0.5)` and `std=(0.5, 0.5, 0.5)`, not ImageNet `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`. Furthermore, extracting `num_classes=1000` projects the rich 1,536-D penultimate pooled features through a 1,000-class ImageNet object vocabulary before disaster classification.
- **Solution**: Use `get_transforms(timm_native=True)` (`(0.5, 0.5, 0.5)` normalization) and `extract_pooled_features_1536` (`forward_head(feat_maps, pre_logits=True)`) alongside `extract_mixed6a_features` (1,088-D intermediate spatial pooled features) for `v3` feature pipelines while retaining backward-compatible feature contracts for legacy 2,036-D checkpoints.
