# DisasterRes-Net

DisasterRes-Net is a disaster response image classification pipeline inspired by Gupta and Roy (2024).

It performs three stages:
1. Data collection from CrisisMMD and web crawlers.
2. Data cleaning, resizing, and train/test split.
3. Feature fusion classification using deep and handcrafted features.

## Project Structure

- step1_collect_data.py: Collects and organizes images into class folders.
- step2_preprocess.py: Removes duplicates, resizes images to 299x299, and creates a 70/30 split.
- step3_classify.py: Extracts InceptionResNetV2 and GLCM features, trains Random Forest, evaluates metrics.
- demo.py: Runs a live single-image prediction demo.
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

pip install tensorflow scikit-learn scikit-image icrawler pillow tqdm joblib matplotlib requests numpy

## How To Run

Run full pipeline:

python run_all.py

Run steps individually:

python step1_collect_data.py
python step2_preprocess.py
python step3_classify.py

Run live demo:

python demo.py

## Expected Outputs

- saved_models contains trained joblib model files.
- results contains confusion matrices and objective metrics.
- split_dataset/train and split_dataset/test contain ready-to-train images.

## Notes

- CrisisMMD is expected to already exist locally.
- Windows path normalization is handled in the scripts.
- If running on CPU only, TensorFlow may be slower during feature extraction.