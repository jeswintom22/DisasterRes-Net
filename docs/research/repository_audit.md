# DisasterRes-Net Repository Audit

## Scope
Traced training, feature extraction, fusion, scaling, checkpoint loading, inference, Flask serialization, visualization, and M2 localization.

## Critical Findings Before Fixes

1. Feature contract was global-name based instead of artifact based. A checkpoint named `df1_df2_glcm_lbp_stats` could still contain a 2030-feature scaler.
2. The 2036 vs 2030 runtime error was caused by LBP schema drift: the code emitted 16 LBP features while the scaler expected a legacy 10-bin LBP vector.
3. The LBP cache key did not include feature schema/version, allowing stale 10-feature caches to be reused by stats-based training.
4. Training, demo inference, hybrid Flask inference, and Step 4 localization contained duplicated feature and saliency logic.
5. Feature names were manually assembled and could diverge from `feature_importances_` length.
6. M2 localization used generic saliency thresholds and did not expose backend choice, polygons, region confidence, or paper-reproduction metadata.
7. Dashboard error handling returned only coarse errors and did not expose model/feature/localization metadata needed for research review.

## Checkpoint Feature Inventory

| artifact | named_pipeline | scaler_features | resolved_pipeline | resolved_dimension | issue |
| --- | --- | --- | --- | --- | --- |
| scaler_damage.joblib | df1_df2_glcm | 2020 | df1_df2_glcm | 2020 |  |
| scaler_damage_df1_df2_glcm_lbp.joblib | df1_df2_glcm_lbp | 2030 | df1_df2_glcm_lbp | 2030 |  |
| scaler_informativeness.joblib | df1_df2_glcm | 2020 | df1_df2_glcm | 2020 |  |
| scaler_informativeness_df1_df2_glcm_lbp.joblib | df1_df2_glcm_lbp | 2030 | df1_df2_glcm_lbp | 2030 |  |
| scaler_informativeness_df1_df2_glcm_lbp_stats.joblib | df1_df2_glcm_lbp_stats | 2030 | df1_df2_glcm_lbp | 2030 | mislabeled: uses df1_df2_glcm_lbp contract |

## Dimension Diagnosis

Current registered feature contracts:

- `df1_df2_glcm`: 2020 features, streams=['cnn_original', 'cnn_saliency', 'glcm']
- `df1_df2_glcm_lbp`: 2030 features, streams=['cnn_original', 'cnn_saliency', 'glcm', 'lbp_hist']
- `df1_df2_glcm_lbp_stats`: 2036 features, streams=['cnn_original', 'cnn_saliency', 'glcm', 'lbp_stats']

The previously observed `2036 features, expected 2030` error is resolved by loading the scaler first, resolving the true feature contract from `n_features_in_`, and extracting either LBP histogram or LBP histogram+statistics accordingly.

## Fixes Applied

- Added `features/feature_contract.py` as the single source of truth for stream dimensions, feature names, and artifact compatibility.
- Updated LBP cache hashing and stale-cache validation in `step3_classify.py`.
- Updated `_load_rf_bundle` to return a resolved `FeaturePipelineSpec` based on scaler dimension.
- Updated hybrid inference to build fused matrices through `build_feature_matrix`/`fuse_feature_streams`.
- Replaced manual feature-importance naming with generated feature names and length validation.
- Added configurable localization backends in `damage_assessment/localization_backends.py`.
- Redesigned DDM/DEM in `damage_assessment/localization.py` with adaptive thresholding, morphology, connected components, region confidence, polygons, and richer DEM terms.
- Added cached-feature ablation framework in `evaluation/ablation.py`.

## Remaining Research Caveats

- Exact SUN implementation details from the original paper are not fully specified; the repo documents its LMS/intensity/gradient + FastICA approximation.
- GradCAM-family backends require PyTorch/timm model loading and are slower than SUN+ICA/OpenCV.
- Pixel-level localization metrics require ground-truth masks. Without masks, the framework reports qualitative/stability metrics instead.