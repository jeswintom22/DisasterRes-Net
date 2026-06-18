# DisasterRes-Net: PyTorch Conversion and GLCM Optimization

This implementation plan outlines the steps to convert the deep learning feature extractor from TensorFlow to PyTorch and optimize the GLCM (Gray-Level Co-occurrence Matrix) extraction process via multiprocessing and disk caching.

> [!WARNING]
> Switching from TensorFlow's `InceptionResNetV2` to PyTorch will require a slightly different setup. PyTorch's native `torchvision` doesn't include the exact `InceptionResNetV2` architecture by default.

## Open Questions
> [!IMPORTANT]
> 1. **Model Architecture**: Since PyTorch's native `torchvision.models` does not include `InceptionResNetV2`, I propose using `timm` (PyTorch Image Models) which contains the exact `inception_resnet_v2` weights pre-trained on ImageNet. Is installing `timm` acceptable to maintain the exact same architecture, or would you prefer a native `torchvision` model like `resnet50` or `inception_v3`?
> 2. **Cache Location**: By default, I will create a `.cache/` directory in the project root to store the `.npz` extracted GLCM features. Is this directory acceptable, or would you prefer `results/`?

## Proposed Changes

### Configuration and Setup
- Install `timm` for exact `InceptionResNetV2` PyTorch implementation (pending your approval).
- Replace TensorFlow imports in `step3_classify.py` with PyTorch (`torch`, `torch.nn`, `torch.utils.data`, `torchvision`).

### `step3_classify.py`

#### [MODIFY] step3_classify.py
- **Remove TensorFlow**: Remove all `tensorflow` and `keras` imports and references.
- **PyTorch Dataset**: Replace `make_tf_dataset` with a custom PyTorch `torch.utils.data.Dataset` class that handles image loading (via PIL), resizing to 299x299, and normalization using standard ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Fine-Tuning Module (`finetune_extractor`)**:
    - Instantiate the PyTorch model (`timm.create_model('inception_resnet_v2', pretrained=True, num_classes=num_classes)`).
    - Freeze all layers except the final classification head (and possibly the top block to match the "top 50 layers" from TF).
    - Use `torch.optim.Adam` and `torch.nn.CrossEntropyLoss`.
    - Run the 5-epoch training loop explicitly (using `model.train()` and `model.eval()`).
    - After fine-tuning, remove the classification head to expose the 1536-dimensional feature vector using `model.forward_features` or similar in `timm`.
- **Deep Feature Extraction (`extract_deep_features`)**:
    - Wrap the PyTorch model in an evaluation context (`with torch.no_grad():`).
    - Loop over the `DataLoader` to extract the embeddings and stack them into a numpy array.
- **GLCM Parallelization & Caching (`extract_all_glcm`)**:
    - Implement a helper to hash the input image paths and their modification times using `hashlib.md5`. This generates a unique `cache_key`.
    - Check if `GLCM_CACHE_DIR/{cache_key}.npz` exists. If so, load and return it immediately, skipping extraction.
    - If no cache exists, use `concurrent.futures.ProcessPoolExecutor` with `os.cpu_count()` workers.
    - Map the `extract_glcm_features` function over all image paths concurrently.
    - Wrap the execution in `tqdm` to show progress.
    - Save the resulting array to `.npz` using `numpy.savez_compressed`.

## Verification Plan

### Automated Tests
- Run `python step3_classify.py` to ensure it completes successfully without syntax or runtime errors.
- Verify that `tqdm` logs show parallel processing speedup during GLCM extraction.
- Run `python step3_classify.py` a second time to ensure the GLCM features load instantly from the `.npz` cache instead of re-extracting.

### Manual Verification
- Review the `results/metrics_*.json` to confirm that PyTorch's fine-tuned `InceptionResNetV2` yields accuracy comparable or superior to the previous TensorFlow implementation.
- Check GPU utilization in Task Manager or via `nvidia-smi` to confirm PyTorch is actively leveraging the GPU.
