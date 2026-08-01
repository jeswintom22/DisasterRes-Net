"""Step 1: collect and organize disaster images.

This script:
1) Organizes CrisisMMD images into raw_dataset classes with unique names.
2) Crawls missing images via icrawler if any class is below target threshold.
3) Filters out tiny / corrupted images.
4) Logs complete data collection metrics and metadata to MLflow.

Windows compatibility:
- Every constructed path is normalized with os.path.normpath.
"""

import csv
import json
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


def npath(*parts: str) -> str:
    """Return a normalized path for Windows-safe joins."""
    return os.path.normpath(os.path.join(*parts))


SAVE_DIR = npath("raw_dataset")
DOWNLOADS_DIR = npath(SAVE_DIR, "_downloads")
MIN_EDGE = 100

DISASTER_CLASSES = ["earthquake", "flood", "hurricane", "wildfire", "landslide", "not_disaster"]


def ensure_dirs() -> None:
    for cls_name in DISASTER_CLASSES:
        try:
            os.makedirs(npath(SAVE_DIR, cls_name), exist_ok=True)
        except OSError as exc:
            print(f"[ERROR] Could not create class directory for '{cls_name}': {exc}")
    try:
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Could not create support directories: {exc}")


def find_crisismmd_dir() -> Optional[str]:
    candidates = [
        npath(SAVE_DIR, "_downloads", "CrisisMMD_v2.0"),
        npath(SAVE_DIR, "CrisisMMD_v2.0"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def is_image_file(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def _safe_copy(src: str, dst: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(npath(src), npath(dst))
        return True
    except (OSError, shutil.Error) as exc:
        print(f"[WARN] Copy failed: '{src}' -> '{dst}'. Reason: {exc}")
        return False


def _pick_existing_file(paths: Sequence[str]) -> Optional[str]:
    for pth in paths:
        if os.path.isfile(pth):
            return pth
    return None


def _resolve_image_path(crisis_dir: str, image_rel: str) -> Optional[str]:
    rel = image_rel.strip().replace("/", os.sep)
    candidate_a = npath(crisis_dir, rel)
    candidate_b = npath(crisis_dir, "data_image", rel)
    if os.path.isfile(candidate_a):
        return candidate_a
    if os.path.isfile(candidate_b):
        return candidate_b
    return None


def _event_to_category(image_rel: str) -> Optional[str]:
    lower_path = image_rel.lower().replace("\\", "/")
    event_map = {
        "earthquake": "earthquake",
        "flood": "flood",
        "hurricane": "hurricane",
        "wildfires": "wildfire",
        "wildfire": "wildfire",
        "landslide": "landslide",
    }
    for key, value in event_map.items():
        if key in lower_path:
            return value
    return None


def _read_tsv_rows(tsv_path: str) -> List[dict]:
    try:
        with open(npath(tsv_path), "r", encoding="utf-8", newline="") as fobj:
            reader = csv.DictReader(fobj, delimiter="\t")
            return list(reader)
    except UnicodeDecodeError:
        try:
            with open(npath(tsv_path), "r", encoding="latin-1", newline="") as fobj:
                reader = csv.DictReader(fobj, delimiter="\t")
                return list(reader)
        except OSError as exc:
            print(f"[ERROR] Failed to read TSV '{tsv_path}': {exc}")
            return []
    except OSError as exc:
        print(f"[ERROR] Failed to read TSV '{tsv_path}': {exc}")
        return []


def _get_first_nonempty(row: dict, columns: Sequence[str]) -> str:
    for col in columns:
        val = row.get(col)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return ""


def organize_crisismmd() -> Dict[str, int]:
    """Copy CrisisMMD images into raw_dataset classes using unique names."""
    stats = {cls_name: 0 for cls_name in DISASTER_CLASSES}
    crisis_dir = find_crisismmd_dir()
    if crisis_dir is None:
        print("[INFO] CrisisMMD folder not found. Expected one of:")
        print(f"  - {npath(SAVE_DIR, '_downloads', 'CrisisMMD_v2.0')}")
        print(f"  - {npath(SAVE_DIR, 'CrisisMMD_v2.0')}")
        print("[INFO] Skipping CrisisMMD organization.")
        return stats

    print(f"[INFO] Found CrisisMMD at: {npath(crisis_dir)}")
    ann_dir = npath(crisis_dir, "annotations")
    if not os.path.isdir(ann_dir):
        print(f"[ERROR] Missing annotations folder: {ann_dir}")
        return stats

    task1_candidates = [
        npath(ann_dir, "task1_all.tsv"),
        npath(ann_dir, "crisismmd_datasplit_agreed_label_task1_all.tsv"),
    ]
    task1_path = _pick_existing_file(task1_candidates)

    tsv_files: List[str] = []
    if task1_path:
        tsv_files.append(task1_path)
    else:
        try:
            for name in sorted(os.listdir(ann_dir)):
                full = npath(ann_dir, name)
                if not os.path.isfile(full):
                    continue
                lname = name.lower()
                if lname.endswith(".tsv") and not lname.startswith("._"):
                    tsv_files.append(full)
        except OSError as exc:
            print(f"[ERROR] Could not list annotations directory '{ann_dir}': {exc}")
            return stats

    if not tsv_files:
        print("[ERROR] No TSV files found for CrisisMMD annotations.")
        return stats

    copied_total = 0
    skipped_total = 0
    for tsv_path in tsv_files:
        print(f"[INFO] Processing TSV: {os.path.basename(tsv_path)}")
        rows = _read_tsv_rows(tsv_path)
        if not rows:
            print("[WARN] TSV had no readable rows.")
            continue

        available_cols = list(rows[0].keys())
        print(f"[INFO] Available columns: {available_cols}")

        required_any = ["image_path", "image", "tweet_image", "image_id"]
        info_any = ["image_info", "label_image", "label", "text_info"]
        if not any(col in available_cols for col in required_any) or not any(col in available_cols for col in info_any):
            print("[ERROR] TSV columns do not match expected schema.")
            continue

        for row in tqdm(rows, desc="  Copying CrisisMMD images"):
            image_rel = _get_first_nonempty(row, ["image_path", "image", "tweet_image"])
            if not image_rel:
                image_id = _get_first_nonempty(row, ["image_id"])
                if image_id:
                    image_rel = image_id + ".jpg"

            info_label = _get_first_nonempty(row, ["image_info", "label_image", "label", "text_info"]).lower()
            info_label = info_label.replace(" ", "_")

            if not image_rel:
                skipped_total += 1
                continue

            src_img = _resolve_image_path(crisis_dir, image_rel)
            if src_img is None:
                skipped_total += 1
                continue

            if info_label == "not_informative":
                target_cls = "not_disaster"
            else:
                target_cls = _event_to_category(image_rel)

            if target_cls is None:
                skipped_total += 1
                continue

            clean_stem = image_rel.strip().replace("/", "_").replace("\\", "_")
            if not clean_stem.lower().endswith((".jpg", ".jpeg", ".png")):
                clean_stem += ".jpg"
            dst_name = f"crisismmd_{clean_stem}"
            dst_path = npath(SAVE_DIR, target_cls, dst_name)

            if os.path.isfile(dst_path):
                stats[target_cls] += 1
                continue

            if _safe_copy(src_img, dst_path):
                stats[target_cls] += 1
                copied_total += 1
            else:
                skipped_total += 1

    print(f"[INFO] CrisisMMD copy summary: new_copied={copied_total}, skipped={skipped_total}")
    return stats


def crawl_missing_classes(target_min: int = 100) -> Dict[str, int]:
    """Use icrawler (BingImageCrawler) to supplement classes below target_min images."""
    crawled_stats: Dict[str, int] = {cls_name: 0 for cls_name in DISASTER_CLASSES}
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("[WARN] icrawler package not installed. Skipping web image crawling.")
        return crawled_stats

    counts = dataset_counts()
    for cls_name in DISASTER_CLASSES:
        current = counts.get(cls_name, 0)
        needed = target_min - current
        if needed <= 0:
            continue

        print(f"[INFO] Class '{cls_name}' has {current} images (target: {target_min}). Crawling {needed} images...")
        save_cat_dir = npath(SAVE_DIR, cls_name)
        os.makedirs(save_cat_dir, exist_ok=True)
        keyword = f"{cls_name} disaster photo" if cls_name != "not_disaster" else "everyday street photo"
        try:
            crawler = BingImageCrawler(storage={"root_dir": save_cat_dir}, log_level=30)
            crawler.crawl(keyword=keyword, max_num=needed)
            new_counts = dataset_counts()
            added = new_counts.get(cls_name, 0) - current
            crawled_stats[cls_name] = max(0, added)
            print(f"[INFO] Crawled {added} new images for '{cls_name}'.")
        except Exception as exc:
            print(f"[WARN] BingImageCrawler failed for '{cls_name}': {exc}")

    return crawled_stats


def remove_tiny_images() -> int:
    print("[INFO] Filtering tiny/corrupted images from raw_dataset...")
    removed = 0
    from PIL import Image

    for cls_name in DISASTER_CLASSES:
        class_dir = npath(SAVE_DIR, cls_name)
        if not os.path.isdir(class_dir):
            continue
        try:
            names = [n for n in os.listdir(class_dir) if is_image_file(n)]
        except OSError as exc:
            print(f"[WARN] Could not read directory '{class_dir}': {exc}")
            continue

        for name in tqdm(names, desc=f"  Filtering {cls_name:14s}", leave=False):
            img_path = npath(class_dir, name)
            try:
                file_size = os.path.getsize(img_path)
                if file_size >= 10 * 1024:
                    continue

                with Image.open(img_path) as img:
                    width, height = img.size
                if width < MIN_EDGE or height < MIN_EDGE:
                    try:
                        os.remove(img_path)
                        removed += 1
                    except OSError as exc:
                        print(f"[WARN] Could not remove tiny image '{img_path}': {exc}")
            except Exception as exc:
                print(f"[WARN] Corrupt or unreadable image skipped '{img_path}': {exc}")
    return removed


def dataset_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cls_name in DISASTER_CLASSES:
        class_dir = npath(SAVE_DIR, cls_name)
        count = 0
        if os.path.isdir(class_dir):
            try:
                for name in os.listdir(class_dir):
                    if is_image_file(name):
                        count += 1
            except OSError as exc:
                print(f"[WARN] Could not list '{class_dir}': {exc}")
        counts[cls_name] = count
    return counts


def main() -> int:
    print("=" * 60)
    print("STEP 1: DATA COLLECTION & MLFLOW LOGGING")
    print("=" * 60)

    ensure_dirs()

    crisis_stats = organize_crisismmd()
    crawled_stats = crawl_missing_classes(target_min=100)
    tiny_removed = remove_tiny_images()

    final_counts = dataset_counts()
    print("\n[SUMMARY] CrisisMMD images processed per class:")
    for cls_name in DISASTER_CLASSES:
        print(f"  {cls_name:14s}: {crisis_stats.get(cls_name, 0)}")

    print("\n[SUMMARY] Crawled images per class:")
    for cls_name in DISASTER_CLASSES:
        print(f"  {cls_name:14s}: {crawled_stats.get(cls_name, 0)}")

    print("\n[SUMMARY] final raw_dataset counts:")
    total = 0
    for cls_name in DISASTER_CLASSES:
        c = final_counts.get(cls_name, 0)
        total += c
        print(f"  {cls_name:14s}: {c}")
    print(f"  total          : {total}")
    print(f"  tiny_removed   : {tiny_removed}")

    if total < 400:
        print("[WARN] Total images are below 400.")

    # MLflow tracking
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("DisasterRes-Net")

        with mlflow.start_run(run_name="step1_data_collection"):
            mlflow.set_tags({"step": "step1_data_collection", "framework": "DisasterRes-Net"})
            mlflow.log_params({
                "min_edge": MIN_EDGE,
                "save_dir": SAVE_DIR,
                "disaster_classes": ",".join(DISASTER_CLASSES),
                "target_min_crawl": 100,
            })
            metrics_dict = {
                "raw_images_total": float(total),
                "tiny_images_removed": float(tiny_removed),
            }
            for cls_name, count in final_counts.items():
                metrics_dict[f"raw_count_{cls_name}"] = float(count)
            for cls_name, cnt in crisis_stats.items():
                metrics_dict[f"crisismmd_count_{cls_name}"] = float(cnt)
            for cls_name, cnt in crawled_stats.items():
                metrics_dict[f"crawled_count_{cls_name}"] = float(cnt)

            mlflow.log_metrics(metrics_dict)

            summary_path = npath(SAVE_DIR, "step1_summary.json")
            with open(summary_path, "w", encoding="utf-8") as fobj:
                json.dump({
                    "final_counts": final_counts,
                    "crisis_stats": crisis_stats,
                    "crawled_stats": crawled_stats,
                    "tiny_removed": tiny_removed,
                    "total": total
                }, fobj, indent=2)
            mlflow.log_artifact(summary_path)
            print("[INFO] Step 1 data collection successfully logged to MLflow experiment 'DisasterRes-Net'.")
    except Exception as exc:
        print(f"[WARN] Could not log Step 1 run to MLflow: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
