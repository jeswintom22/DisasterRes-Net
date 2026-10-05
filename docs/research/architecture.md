# DisasterRes-Net Research Architecture

## Shared Feature Pipeline
All training, evaluation, inference, and Flask prediction paths use feature contracts from `features/feature_contract.py`.

Supported configurations:

- `cnn`: 2000 features, streams=['cnn_original', 'cnn_saliency']
- `glcm`: 20 features, streams=['glcm']
- `lbp`: 16 features, streams=['lbp_stats']
- `cnn_lbp`: 2016 features, streams=['cnn_original', 'cnn_saliency', 'lbp_stats']
- `cnn_glcm`: 2020 features, streams=['cnn_original', 'cnn_saliency', 'glcm']
- `glcm_lbp`: 36 features, streams=['glcm', 'lbp_stats']
- `cnn_glcm_lbp`: 2036 features, streams=['cnn_original', 'cnn_saliency', 'glcm', 'lbp_stats']
- `df1_df2_glcm`: 2020 features, streams=['cnn_original', 'cnn_saliency', 'glcm']
- `df1_df2_glcm_lbp`: 2030 features, streams=['cnn_original', 'cnn_saliency', 'glcm', 'lbp_hist']
- `df1_df2_glcm_lbp_stats`: 2036 features, streams=['cnn_original', 'cnn_saliency', 'glcm', 'lbp_stats']

## M1 Prediction
Image inputs are transformed into CNN original/saliency streams, GLCM descriptors, and LBP descriptors. The selected feature spec determines the fused vector and generated feature names. The scaler dimension is validated before classifier inference.

## M2 Localization
Backends: `sun_ica`, `opencv`, `gradcam`, `gradcam_plus_plus`, `scorecam`. Each backend produces a normalized activation map consumed by the same DDM/DEM analyzer.

## DEM Equation
DEM = 100 * severity_weight * (0.22A + 0.18D + 0.13L + 0.09R + 0.10C + 0.08K + 0.07B + 0.06T + 0.04Q + 0.03M), clipped to [0, 100].

Where A=damage area ratio, D=damage density, L=largest-region ratio, R=average-region-size term, C=region-count term, K=1-compactness, B=boundary complexity, T=texture entropy, Q=localization confidence, M=average activation.