# Efficiency Notes

## Inference feature extractor reuse

The fused classifier still requires two CNN feature streams:

- `DF1` / `cnn_original`: InceptionResNetV2 features from the saliency-enhanced RGB image.
- `DF2` / `cnn_saliency`: InceptionResNetV2 features from the saliency image stream.

These streams do not require two independent frozen InceptionResNetV2 model
instances. Both streams use the same pretrained feature extractor with different
input images. `_load_shared_feature_models()` now caches one frozen model and
returns that same instance for both stream positions, preserving the existing
caller API while avoiding duplicate model memory and startup cost.

## Per-request feature reuse

The dashboard inference path predicts both objectives for the same uploaded
image. CNN, GLCM, and LBP features are image-level values, not objective-level
values, so they are now computed once per request and reused for each objective's
artifact-specific feature contract.

The dashboard path also sends the enhanced RGB image and saliency image through
the shared CNN as a two-image batch. This removes repeated objective-level CNN
work, reduces the two CNN streams to one forward pass per request, and avoids
repeated GLCM extraction plus an extra LBP disk read when both `informativeness`
and `damage` predictions are requested.

## Dashboard latency updates

The dashboard now keeps the uploaded image in memory through saliency, CNN, GLCM,
and RandomForest inference. It no longer writes temporary JPEG files just so the
feature extractors can load them again. The RandomForest, scaler, and label
encoder bundles are cached with `lru_cache`, so repeated requests reuse the
large joblib artifacts after the first load.

Single-image web inference also suppresses progress bars, which keeps the Flask
console quieter and avoids per-request terminal rendering overhead.

## Saliency and LBP reuse

The M1 pipeline already computes the saliency map and LBP texture descriptors.
M2 damage localization now accepts those precomputed values from the Flask route
instead of recomputing LBP entropy for the same image. The dashboard defaults to
the `provided` localization backend, which reuses the M1 saliency map for fast
interactive use while leaving SUN+ICA selectable for research comparison.

## Damage severity calibration

The dashboard previously passed the informativeness output, such as
`informative`, into DEM scoring as if it were the disaster type. That skipped
disaster-specific DEM weighting for images like `earthquakeP.jpg` and could
lower the displayed severity. The app now resolves disaster context from an
explicit form field first, then filename keywords, then `unknown`.

No retraining was performed for this fix. A conservative underclassification
guard now documents and applies a minimum DEM score only when the RandomForest
damage classifier is confident that the image is at least a higher damage tier.
For example, a confident RF `severe` label can lift a below-moderate DEM to the
moderate boundary, but it does not blindly force the DEM display to severe.
