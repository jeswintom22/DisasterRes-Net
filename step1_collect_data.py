"""Step 1: collect and organize disaster images.

This script:
1) Organizes already extracted CrisisMMD images into raw_dataset classes.
2) Optionally supplements data via icrawler (Google and Bing).

Windows compatibility:
- Every constructed path is normalized with os.path.normpath.
- Temporary crawler directories are created explicitly to avoid WinError 3.
"""

import csv
import os
import shutil
import time
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


def npath(*parts: str) -> str:
    """Return a normalized path for Windows-safe joins."""
    return os.path.normpath(os.path.join(*parts))


SAVE_DIR = npath("raw_dataset")
DOWNLOADS_DIR = npath(SAVE_DIR, "_downloads")
TMP_DIR = npath(SAVE_DIR, "_tmp")
IMAGES_PER_KW = 80
MIN_EDGE = 100

DISASTER_CLASSES = ["earthquake", "flood", "hurricane", "wildfire", "landslide", "not_disaster"]

KEYWORDS: Dict[str, List[str]] = {
    "earthquake": [
        "earthquake building damage",
        "collapsed building earthquake",
        "earthquake rescue",
    ],
    "flood": [
        "flood disaster damage",
        "flooded roads and homes",
        "flood rescue operation",
    ],
    "hurricane": [
        "hurricane damage destruction",
        "cyclone aftermath",
        "storm surge destruction",
    ],
    "wildfire": [
        "wildfire burning homes",
        "forest fire destruction",
        "wildfire smoke evacuation",
    ],
    "landslide": [
        "landslide road damage",
        "mudslide disaster",
    ],
    "not_disaster": [
        "normal city street",
        "people in park sunny day",
        "clean residential neighborhood",
    ],
}


def ensure_dirs() -> None:
    for cls_name in DISASTER_CLASSES:
        try:
            os.makedirs(npath(SAVE_DIR, cls_name), exist_ok=True)
        except OSError as exc:
            print(f"[ERROR] Could not create class directory for '{cls_name}': {exc}")
    try:
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        os.makedirs(TMP_DIR, exist_ok=True)
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


def _safe_move(src: str, dst: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(npath(src), npath(dst))
        return True
    except (OSError, shutil.Error) as exc:
        print(f"[WARN] Move failed: '{src}' -> '{dst}'. Reason: {exc}")
        return False


def _safe_remove_tree(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(npath(path), ignore_errors=True)
    except OSError as exc:
        print(f"[WARN] Could not remove temporary directory '{path}': {exc}")


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
    """Copy CrisisMMD images into raw_dataset classes.

    Logic:
    - If image informativeness is not_informative -> not_disaster
    - Else map event path to disaster class (earthquake/flood/...)
    """
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
            print("[ERROR] Expected one image path column from: image_path, image, tweet_image, image_id")
            print("[ERROR] Expected one label column from: image_info, label_image, label, text_info")
            print(f"[ERROR] Found columns: {available_cols}")
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

            dst_name = os.path.basename(src_img)
            dst_path = npath(SAVE_DIR, target_cls, dst_name)
            if os.path.isfile(dst_path):
                continue

            if _safe_copy(src_img, dst_path):
                stats[target_cls] += 1
                copied_total += 1
            else:
                skipped_total += 1

    print(f"[INFO] CrisisMMD copy summary: copied={copied_total}, skipped={skipped_total}")
    return stats


def _move_crawler_images(src_dir: str, dst_dir: str, prefix: str) -> int:
    moved = 0
    if not os.path.isdir(src_dir):
        return moved
    try:
        names = os.listdir(src_dir)
    except OSError as exc:
        print(f"[WARN] Could not list temporary crawler directory '{src_dir}': {exc}")
        return moved

    for name in names:
        if not is_image_file(name):
            continue
        src = npath(src_dir, name)
        dst = npath(dst_dir, f"{prefix}{name}")
        if _safe_move(src, dst):
            moved += 1
    _safe_remove_tree(src_dir)
    return moved


def collect_via_icrawler() -> Dict[str, int]:
    """Collect extra images via Google and Bing crawlers."""
    stats = {cls_name: 0 for cls_name in DISASTER_CLASSES}
    try:
        from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
    except ImportError:
        print("[WARN] icrawler not installed. Install with: pip install icrawler")
        return stats

    for category, query_list in KEYWORDS.items():
        dst_dir = npath(SAVE_DIR, category)
        try:
            os.makedirs(dst_dir, exist_ok=True)
        except OSError as exc:
            print(f"[ERROR] Could not create destination directory '{dst_dir}': {exc}")
            continue

        print(f"[INFO] Collecting category: {category}")
        for query in query_list:
            google_tmp = npath(TMP_DIR, "google", category)
            bing_tmp = npath(TMP_DIR, "bing", category)
            try:
                os.makedirs(google_tmp, exist_ok=True)
                os.makedirs(bing_tmp, exist_ok=True)
            except OSError as exc:
                print(f"[ERROR] Could not create temp folders for '{category}': {exc}")
                continue

            try:
                google = GoogleImageCrawler(storage={"root_dir": npath(google_tmp)}, log_level=50)
                google.crawl(keyword=query, max_num=IMAGES_PER_KW // 2, file_idx_offset="auto")
                moved_g = _move_crawler_images(google_tmp, dst_dir, f"g_{category}_")
                stats[category] += moved_g
                print(f"  Google '{query}': moved {moved_g}")
            except Exception as exc:
                print(f"[WARN] Google crawl failed for '{query}': {exc}")
                _safe_remove_tree(google_tmp)

            try:
                bing = BingImageCrawler(storage={"root_dir": npath(bing_tmp)}, log_level=50)
                bing.crawl(keyword=query, max_num=IMAGES_PER_KW // 2, file_idx_offset="auto")
                moved_b = _move_crawler_images(bing_tmp, dst_dir, f"b_{category}_")
                stats[category] += moved_b
                print(f"  Bing   '{query}': moved {moved_b}")
            except Exception as exc:
                print(f"[WARN] Bing crawl failed for '{query}': {exc}")
                _safe_remove_tree(bing_tmp)

            time.sleep(1)
    return stats


def remove_tiny_images() -> int:
    removed = 0
    for cls_name in DISASTER_CLASSES:
        class_dir = npath(SAVE_DIR, cls_name)
        if not os.path.isdir(class_dir):
            continue
        try:
            names = os.listdir(class_dir)
        except OSError as exc:
            print(f"[WARN] Could not read directory '{class_dir}': {exc}")
            continue

        for name in names:
            if not is_image_file(name):
                continue
            img_path = npath(class_dir, name)
            try:
                from PIL import Image

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
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)

    ensure_dirs()

    crisis_stats = organize_crisismmd()
    crawl_stats = collect_via_icrawler()
    tiny_removed = remove_tiny_images()

    final_counts = dataset_counts()
    print("\n[SUMMARY] CrisisMMD copied per class:")
    for cls_name in DISASTER_CLASSES:
        print(f"  {cls_name:14s}: {crisis_stats.get(cls_name, 0)}")

    print("\n[SUMMARY] icrawler added per class:")
    for cls_name in DISASTER_CLASSES:
        print(f"  {cls_name:14s}: {crawl_stats.get(cls_name, 0)}")

    print("\n[SUMMARY] final raw_dataset counts:")
    total = 0
    for cls_name in DISASTER_CLASSES:
        c = final_counts.get(cls_name, 0)
        total += c
        print(f"  {cls_name:14s}: {c}")
    print(f"  total          : {total}")
    print(f"  tiny_removed   : {tiny_removed}")

    if total < 400:
        print("[WARN] Total images are below 400. Add more web collection if needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
