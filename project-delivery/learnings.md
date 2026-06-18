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
