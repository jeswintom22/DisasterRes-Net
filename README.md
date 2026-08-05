# DisasterRes-Net

DisasterRes-Net is a disaster response image classification pipeline inspired by Gupta and Roy (2024).

It performs three stages:
1. Data collection from CrisisMMD and web crawlers.
2. Data cleaning, resizing, and train/test split.
3. Feature fusion classification using deep and handcrafted features.

## Project Structure

- step1_collect_data.py: Collects and organizes images into class folders.
- step2_preprocess.py: Removes duplicates, resizes images to 299x299, and creates a 70/30 split.
- step3_classify.py: Uses PyTorch (`timm` + `torchvision`) to extract deep features, fuses them with GLCM, trains Random Forest, evaluates metrics, and logs MLflow runs.
- demo.py: Runs a live single-image prediction demo using the saved fused-feature inference path.
- run_all.py: Executes step1 -> step2 -> validation -> step3 as a full pipeline.

Primary data folders:
- raw_dataset: Source images by category.
- clean_dataset: Deduplicated and resized images.
- split_dataset: train and test split.
- saved_models: Trained model artifacts.
- results: Metrics and confusion matrices.

## Requirements

Recommended Python: 3.10 (Windows).

Install dependencies:

pip install torch torchvision timm scikit-learn scikit-image icrawler pillow tqdm joblib matplotlib requests numpy mlflow

## How To Run

Run full pipeline:

python run_all.py

Run steps individually:

python step1_collect_data.py
python step2_preprocess.py
python step3_classify.py

Run live demo:

python demo.py

Run the research ablation study from cached feature matrices:

python -m evaluation.ablation

Generate audit, architecture, validation, and final project reports:

python -m reports.generate_reports

Run the Flask research dashboard:

python app.py

## Live Incident Monitoring

The dashboard can query the configured live disaster sources, then collect and assess social imagery for an incident selected from the **Current incidents** list.

1. Create a dedicated X account for the project and copy `.env.example` to `.env`.
2. Set `TWIKIT_USERNAME`, `TWIKIT_EMAIL`, and `TWIKIT_PASSWORD` in `.env`.
3. Restart `python app.py` after changing `.env`; credentials are read when the Flask process starts.
4. Open `http://127.0.0.1:5000`, select **Scan Now**, then select an incident card.

The incident workspace advances through Locate, Search, Validate, and Classify. A valid image card opens the full hybrid-model analysis with saliency, LBP, DDM, damage mask, DEM, and emergency recommendations.

If the workspace says social image search needs Twikit credentials, verify all three variables in `.env` are non-empty and restart the dashboard. The agent saves its X session in `.cache/twikit_cookies.json` after a successful login and observes the configured search cap and cooldown.

If the workspace reports `CERTIFICATE_VERIFY_FAILED`, the HTTPS certificate chain was rejected before X login. Upgrade the local certificate bundle with `python -m pip install --upgrade certifi requests httpx twikit`, restart the dashboard, and retry. On a managed school or company network, ask the network administrator to install the proxy's root certificate in the Windows trusted-root store. Do not disable TLS certificate verification.

## Expected Outputs

- saved_models contains Random Forest models, label encoders, scalers, and optional PyTorch checkpoints.
- results contains confusion matrices, objective metrics, feature-importance reports, and inference specs.
- split_dataset/train and split_dataset/test contain ready-to-train images.

## Notes

- CrisisMMD is expected to already exist locally.
- Windows path normalization is handled in the scripts.
- If running on CPU only, PyTorch feature extraction will be slower.
- Optional direct PyTorch classifier training can be enabled with `DISASTERRES_TORCH_TRAIN=1`.
## Hybrid Research Extension

The project now includes a modular research-grade hybrid pipeline:

- `preprocessing/lbp.py`: normalized LBP histogram, texture map, and statistical descriptors.
- `preprocessing/saliency.py`: saliency detection and attention-weighted image generation.
- `fusion/feature_fusion.py`: CNN + handcrafted feature concatenation metadata.
- `models/hybrid_pipeline.py`: saliency-attended CNN inference plus LBP/GLCM fusion with checkpoint fallback.
- `damage_assessment/localization.py`: M2 damage mask, connected regions, DDM, DEM, and impact estimates.
- `HYBRID_RESEARCH_ARCHITECTURE.md`: architectural notes and checkpoint compatibility details.

Existing `df1_df2_glcm_lbp` checkpoints remain usable. Retraining `step3_classify.py` creates the newer `df1_df2_glcm_lbp_stats` artifacts with LBP statistical descriptors included in the fused classifier input.

The loader resolves feature contracts from each scaler's `n_features_in_`, so mislabeled or legacy checkpoints fail with informative messages instead of silent 2036/2030 feature mismatches.

