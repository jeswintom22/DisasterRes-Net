"""Run full DisasterRes-Net pipeline: step1 -> step2 -> validation -> step3."""

import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError


def npath(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


ROOT_DIR = npath(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = npath(ROOT_DIR, "raw_dataset")
SPLIT_DIR = npath(ROOT_DIR, "split_dataset")
RESULT_DIR = npath(ROOT_DIR, "results")
EXPECTED_CLASSES = ["earthquake", "flood", "hurricane", "wildfire", "landslide", "not_disaster"]
EXPECTED_IMG_SIZE = (299, 299)
MIN_IMAGES_PER_CLASS = 50


def run_step(script_name: str) -> bool:
    script_path = npath(ROOT_DIR, script_name)
    if not os.path.isfile(script_path):
        print(f"[ERROR] Missing script: {script_path}")
        return False

    print("\n" + "=" * 80)
    print(f"RUNNING {script_name}")
    print("=" * 80)
    try:
        result = subprocess.run([sys.executable, script_path], cwd=ROOT_DIR, check=False)
    except OSError as exc:
        print(f"[ERROR] Failed to start '{script_name}': {exc}")
        print("[HINT] Ensure Python is installed and file permissions are valid.")
        return False

    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed with exit code {result.returncode}.")
        print("[HINT] Scroll logs above for exact failure details.")
        return False

    print(f"[OK] {script_name} completed successfully.")
    return True


def count_images_in_dir(path: str) -> int:
    count = 0
    if not os.path.isdir(path):
        return count
    try:
        for name in os.listdir(path):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1
    except OSError as exc:
        print(f"[WARN] Could not list image directory '{path}': {exc}")
    return count


def raw_dataset_stats() -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for cls_name in EXPECTED_CLASSES:
        stats[cls_name] = count_images_in_dir(npath(RAW_DIR, cls_name))
    return stats


def split_dataset_stats() -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {"train": {}, "test": {}}
    for split_name in ("train", "test"):
        split_root = npath(SPLIT_DIR, split_name)
        if not os.path.isdir(split_root):
            continue
        try:
            for cls_name in os.listdir(split_root):
                cls_dir = npath(split_root, cls_name)
                if os.path.isdir(cls_dir):
                    stats[split_name][cls_name] = count_images_in_dir(cls_dir)
        except OSError as exc:
            print(f"[WARN] Could not list split directory '{split_root}': {exc}")
    return stats


def validate_before_step3() -> Tuple[bool, List[str]]:
    """Validate dataset quality before model training."""
    issues: List[str] = []

    if not os.path.isdir(SPLIT_DIR):
        issues.append(f"split_dataset missing at {SPLIT_DIR}. Run step2_preprocess.py first.")
        return False, issues

    split_stats = split_dataset_stats()

    for cls_name in EXPECTED_CLASSES:
        total_cls = split_stats.get("train", {}).get(cls_name, 0) + split_stats.get("test", {}).get(cls_name, 0)
        if total_cls < MIN_IMAGES_PER_CLASS:
            issues.append(
                f"Class '{cls_name}' has {total_cls} images in split_dataset; need at least {MIN_IMAGES_PER_CLASS}."
            )

    for split_name in ("train", "test"):
        split_root = npath(SPLIT_DIR, split_name)
        if not os.path.isdir(split_root):
            issues.append(f"Missing split folder: {split_root}")
            continue

        for cls_name in EXPECTED_CLASSES:
            cls_dir = npath(split_root, cls_name)
            if not os.path.isdir(cls_dir):
                issues.append(f"Missing class folder: {cls_dir}")
                continue

            img_count = count_images_in_dir(cls_dir)
            if img_count == 0:
                issues.append(f"Empty class folder: {cls_dir}")
                continue

            checked = 0
            try:
                names = os.listdir(cls_dir)
            except OSError as exc:
                issues.append(f"Cannot list class folder '{cls_dir}': {exc}")
                continue

            for fname in names:
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = npath(cls_dir, fname)
                checked += 1
                try:
                    with Image.open(img_path) as img:
                        if img.size != EXPECTED_IMG_SIZE:
                            issues.append(
                                f"Invalid size in {img_path}: got {img.size}, expected {EXPECTED_IMG_SIZE}. Rerun step2."
                            )
                            break
                except UnidentifiedImageError as exc:
                    issues.append(f"Corrupt image in split dataset: {img_path}. Reason: {exc}")
                except OSError as exc:
                    issues.append(f"Cannot open image '{img_path}': {exc}")

            if checked == 0:
                issues.append(f"No images found in folder: {cls_dir}")

    return len(issues) == 0, issues


def load_accuracy_results() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for objective in ("informativeness", "damage"):
        metrics_path = npath(RESULT_DIR, f"metrics_{objective}.json")
        if not os.path.isfile(metrics_path):
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as fobj:
                data = json.load(fobj)
            out[objective] = float(data.get("accuracy", 0.0))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"[WARN] Could not read metrics file '{metrics_path}': {exc}")
    return out


def print_progress_summary(step_name: str) -> None:
    print("\n" + "-" * 80)
    print(f"SUMMARY AFTER {step_name}")
    print("-" * 80)

    raw_stats = raw_dataset_stats()
    print("raw_dataset counts:")
    total_raw = 0
    for cls_name in EXPECTED_CLASSES:
        count = raw_stats.get(cls_name, 0)
        total_raw += count
        print(f"  {cls_name:14s}: {count}")
    print(f"  total          : {total_raw}")

    split_stats = split_dataset_stats()
    train_total = sum(split_stats.get("train", {}).values())
    test_total = sum(split_stats.get("test", {}).values())
    print("split_dataset counts:")
    print(f"  train total    : {train_total}")
    print(f"  test total     : {test_total}")

    accs = load_accuracy_results()
    if accs:
        print("accuracy results:")
        for objective, score in accs.items():
            print(f"  {objective:14s}: {score * 100:.2f}%")


def main() -> int:
    print("=" * 80)
    print("DISASTERRES-NET PIPELINE RUNNER")
    print("=" * 80)

    if not run_step("step1_collect_data.py"):
        return 1
    print_progress_summary("STEP 1")

    if not run_step("step2_preprocess.py"):
        return 1
    print_progress_summary("STEP 2")

    ok, issues = validate_before_step3()
    if not ok:
        print("\n[ERROR] Validation before step3 failed. Fix the following issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("[HINT] Re-run step1 and step2 after fixing data issues.")
        return 1

    print("\n[OK] Validation passed: sufficient images, correct 299x299 sizes, and no empty split folders.")

    if not run_step("step3_classify.py"):
        return 1
    print_progress_summary("STEP 3")

    print("\n[DONE] Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
