"""Step 4: M2 Module - Damage Distribution Map (DDM) & Damage Evaluation Metric (DEM).

Paper: Gupta & Roy (2025) - M2 module
This module generates saliency-based damage heatmaps and scores.

Outputs:
  - DDM (Damage Distribution Map): saliency heatmap overlaid on image (PNG)
  - DEM (Damage Evaluation Metric): scalar score (mean saliency)
  - Classification: "little_or_none" (DEM < 0.25), "mild" (0.25 ≤ DEM ≤ 0.50), "severe" (DEM > 0.50)
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import json

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except Exception as exc:
    plt = None
    cm = None
    print(f"[WARN] matplotlib import failed: {exc}. DDM visualization disabled.")

try:
    from scipy.ndimage import laplace
except Exception as exc:
    print(f"[ERROR] scipy import failed: {exc}")

_SPLIT_DIR = os.path.normpath(os.path.join("split_dataset"))
_RESULT_DIR = os.path.normpath(os.path.join("results"))
_DDM_DIR = os.path.normpath(os.path.join("results", "ddm_heatmaps"))
_IMG_SIZE = (299, 299)
_IMG_EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(_RESULT_DIR, exist_ok=True)
os.makedirs(_DDM_DIR, exist_ok=True)


def rgb_to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale."""
    if len(img_rgb.shape) == 2:
        return img_rgb.astype(np.float32) / 255.0
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    return (0.299*r + 0.587*g + 0.114*b).astype(np.float32) / 255.0


def compute_saliency_map(img_rgb: np.ndarray) -> np.ndarray:
    """Compute saliency map using Laplacian-based edge detection.
    
    Approximation of the SUN model from the paper.
    Returns (H, W) saliency in [0, 1].
    """
    try:
        img_gray = rgb_to_grayscale(img_rgb)
        sal = np.abs(laplace(img_gray))
        
        sal_min, sal_max = sal.min(), sal.max()
        if sal_max > sal_min:
            sal = (sal - sal_min) / (sal_max - sal_min)
        else:
            sal = np.ones_like(sal) * 0.5
        
        return sal
    except Exception as exc:
        print(f"[WARN] Saliency computation failed: {exc}")
        return np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32) * 0.5


def compute_dem_score(saliency_map: np.ndarray) -> float:
    """Compute Damage Evaluation Metric (DEM) = mean saliency."""
    return float(np.mean(saliency_map))


def dem_to_damage_class(dem: float) -> str:
    """Classify DEM score into damage severity category.
    
    Paper thresholds: d1=0.25, d2=0.50
    """
    if dem < 0.25:
        return "little_or_none"
    elif dem <= 0.50:
        return "mild"
    else:
        return "severe"


def create_ddm_heatmap(img_rgb: np.ndarray, saliency_map: np.ndarray, 
                       output_path: str, cmap: str = "hot") -> bool:
    """Create and save DDM visualization (saliency overlaid on image).
    
    Args:
        img_rgb: Original image (H, W, 3) in [0, 255]
        saliency_map: Saliency map (H, W) in [0, 1]
        output_path: Path to save DDM PNG
        cmap: Colormap name (e.g., "hot", "jet", "viridis")
    
    Returns:
        True if successful, False otherwise
    """
    if plt is None or cm is None:
        return False
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(img_rgb.astype(np.uint8))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Saliency heatmap
        im = axes[1].imshow(saliency_map, cmap=cmap)
        axes[1].set_title("Damage Distribution Map (DDM)")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"[WARN] Failed to save DDM heatmap to '{output_path}': {exc}")
        return False


def process_image(image_path: str) -> Optional[Dict[str, float]]:
    """Process single image: extract saliency, compute DEM, create DDM.
    
    Returns:
        Dict with keys: 'dem', 'damage_class', 'saliency_map' (array)
        None if processing fails
    """
    try:
        with Image.open(image_path) as img:
            img_rgb = np.asarray(img.convert('RGB'), dtype=np.uint8)
            img_rgb = np.asarray(Image.fromarray(img_rgb).resize(_IMG_SIZE, Image.Resampling.LANCZOS))
    except UnidentifiedImageError as exc:
        print(f"[WARN] Corrupt image: '{image_path}'. Reason: {exc}")
        return None
    except OSError as exc:
        print(f"[WARN] Could not read '{image_path}': {exc}")
        return None
    
    # Compute saliency and DEM
    saliency = compute_saliency_map(img_rgb)
    dem = compute_dem_score(saliency)
    damage_class = dem_to_damage_class(dem)
    
    return {
        'dem': dem,
        'damage_class': damage_class,
        'saliency_map': saliency,
    }


def process_dataset_split(split: str = "test") -> Dict[str, Dict]:
    """Process all images in a dataset split and generate DDM heatmaps.
    
    Args:
        split: "train" or "test"
    
    Returns:
        Dict mapping image_path -> {dem, damage_class, heatmap_path}
    """
    split_dir = os.path.normpath(os.path.join(_SPLIT_DIR, split))
    if not os.path.isdir(split_dir):
        print(f"[ERROR] Split folder not found: {split_dir}")
        return {}
    
    print(f"\n[STEP] Processing {split.upper()} split for M2 module")
    results = {}
    
    # Iterate over all class folders
    for class_folder in sorted(os.listdir(split_dir)):
        class_path = os.path.normpath(os.path.join(split_dir, class_folder))
        if not os.path.isdir(class_path):
            continue
        
        # Get all image files in this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(_IMG_EXTS)]
        
        print(f"  Processing class '{class_folder}' ({len(image_files)} images)...")
        
        for img_file in tqdm(image_files, desc=f"    {class_folder}"):
            img_path = os.path.normpath(os.path.join(class_path, img_file))
            
            proc = process_image(img_path)
            if proc is None:
                continue
            
            # Generate DDM heatmap filename
            rel_path = os.path.relpath(img_path, _SPLIT_DIR)
            heatmap_name = rel_path.replace(os.sep, "_").replace(".", "_") + "_ddm.png"
            heatmap_path = os.path.normpath(os.path.join(_DDM_DIR, heatmap_name))
            
            # Create and save DDM visualization
            with Image.open(img_path) as img:
                img_rgb = np.asarray(img.convert('RGB'), dtype=np.uint8)
            create_ddm_heatmap(img_rgb, proc['saliency_map'], heatmap_path)
            
            results[img_path] = {
                'dem': proc['dem'],
                'damage_class': proc['damage_class'],
                'heatmap_path': heatmap_path,
            }
    
    return results


def compute_m2_metrics(results_dict: Dict[str, Dict]) -> Dict[str, float]:
    """Compute M2 module metrics from processed images.
    
    Args:
        results_dict: Dict mapping image_path -> {dem, damage_class, ...}
    
    Returns:
        Metrics dict with DEM distribution and classification stats
    """
    if not results_dict:
        return {}
    
    dems = [r['dem'] for r in results_dict.values()]
    damage_classes = [r['damage_class'] for r in results_dict.values()]
    
    little_count = damage_classes.count('little_or_none')
    mild_count = damage_classes.count('mild')
    severe_count = damage_classes.count('severe')
    
    return {
        'total_images': len(results_dict),
        'dem_mean': float(np.mean(dems)),
        'dem_std': float(np.std(dems)),
        'dem_min': float(np.min(dems)),
        'dem_max': float(np.max(dems)),
        'little_or_none_count': int(little_count),
        'mild_count': int(mild_count),
        'severe_count': int(severe_count),
        'little_or_none_pct': float(100 * little_count / len(results_dict)),
        'mild_pct': float(100 * mild_count / len(results_dict)),
        'severe_pct': float(100 * severe_count / len(results_dict)),
    }


def save_m2_report(split_results: Dict[str, Dict], split: str = "test") -> str:
    """Save M2 module report (metrics + DEM distribution).
    
    Returns:
        Path to saved report JSON
    """
    metrics = compute_m2_metrics(split_results)
    report = {
        'module': 'M2',
        'split': split,
        'metrics': metrics,
        'note': 'DEM thresholds: d1=0.25 (little_or_none), d2=0.50 (mild/severe)',
        'saliency_method': 'Laplacian-based approximation of SUN model',
    }
    
    report_path = os.path.normpath(os.path.join(_RESULT_DIR, f"m2_report_{split}.json"))
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] M2 report saved: {report_path}")
    except OSError as exc:
        print(f"[WARN] Could not save M2 report: {exc}")
    
    return report_path


def main():
    """Main entry point: process all splits and generate M2 outputs."""
    print("\n" + "=" * 70)
    print("STEP 4 - M2 Module: Damage Distribution Map (DDM) & DEM Scoring")
    print("=" * 70)
    
    # Process test split
    test_results = process_dataset_split(split="test")
    if test_results:
        save_m2_report(test_results, split="test")
        print(f"[RESULT] Processed {len(test_results)} test images")
        print(f"[INFO] DDM heatmaps saved to: {_DDM_DIR}")
    else:
        print("[WARN] No test images processed")
    
    # Process train split (optional, slower)
    print("\n[STEP] Processing TRAIN split (optional, this may take a while)...")
    response = input("Process train split? (y/n): ").strip().lower()
    if response == 'y':
        train_results = process_dataset_split(split="train")
        if train_results:
            save_m2_report(train_results, split="train")
            print(f"[RESULT] Processed {len(train_results)} train images")
        else:
            print("[WARN] No train images processed")
    
    print("\n" + "=" * 70)
    print("STEP 4 Complete - M2 module outputs saved")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
