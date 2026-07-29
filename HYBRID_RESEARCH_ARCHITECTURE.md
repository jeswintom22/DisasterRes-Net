# Hybrid DisasterRes-Net Research Architecture

This refactor turns LBP and saliency from dashboard-only visuals into active inference components.

## M1 Hybrid Prediction Pipeline

1. RGB image is passed through `preprocessing.saliency.SaliencyAttention`.
2. Saliency creates an attention mask and enhanced RGB image.
3. The enhanced image enters the InceptionResNetV2 CNN feature stream.
4. A saliency image is also used as the second CNN stream for compatibility with the original paper-aligned DF2 path.
5. `preprocessing.lbp.LBPFeatureExtractor` computes grayscale LBP, a normalized histogram, and statistical descriptors.
6. `fusion.feature_fusion.concatenate_features` concatenates CNN, saliency-CNN, GLCM, and LBP streams.
7. The saved StandardScaler and RandomForest classifier produce prediction, confidence, class distribution, and feature metadata.

## Checkpoint Compatibility

Existing checkpoints named `df1_df2_glcm_lbp` are preserved. They use the legacy 10-bin LBP histogram feature vector.

New training uses `FEATURE_PIPELINE = df1_df2_glcm_lbp_stats`, which expands the LBP stream to histogram plus six statistics: mean, variance, entropy, energy, dominant bin, and uniformity. Retrain `step3_classify.py` to create the new stats-rich checkpoints.

## M2 Damage Localization

`damage_assessment.localization.DamageLocalizationAnalyzer` implements the second stage:

1. Saliency detection.
2. Percentile thresholding into a binary damage mask.
3. Morphological opening/closing to suppress noise.
4. Connected-component region detection.
5. Damage Distribution Map generation with heat overlay, region boxes, and centroids.
6. Damage Evaluation Metric from affected area, saliency density, region count, compactness, and disaster severity weighting.

DEM is normalized to 0-100:

- 0-20 Minimal
- 21-40 Mild
- 41-60 Moderate
- 61-80 Severe
- 81-100 Critical

## Frontend Outputs

The Flask dashboard now displays synchronized outputs for original image, saliency, LBP texture map, damage mask, DDM, GradCAM-compatible overlay, predictions, confidence distribution, DEM, affected area, connected regions, texture statistics, fusion metadata, and emergency recommendations.
