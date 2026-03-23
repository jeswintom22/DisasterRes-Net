"""Step 2: clean, deduplicate, resize, and split dataset."""

import hashlib
import os
import shutil
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def npath(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


RAW_DIR = npath("raw_dataset")
CLEAN_DIR = npath("clean_dataset")
SPLIT_DIR = npath("split_dataset")
IMG_SIZE = (299, 299)
TEST_SIZE = 0.30
RANDOM_SEED = 42

IMG_EXTS = (".jpg", ".jpeg", ".png")


def safe_remove_dir(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(npath(path), ignore_errors=True)
    except OSError as exc:
        print(f"[WARN] Could not remove directory '{path}': {exc}")


def get_md5(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    """Return MD5 hash in streaming mode to avoid high memory use."""
    digest = hashlib.md5()
    try:
        with open(npath(filepath), "rb") as fobj:
            while True:
                data = fobj.read(chunk_size)
                if not data:
                    break
                digest.update(data)
        return digest.hexdigest()
    except OSError as exc:
        print(f"[WARN] Could not hash file '{filepath}': {exc}")
        return ""


def _list_raw_categories() -> List[str]:
    categories: List[str] = []
    skip_dirs = {"_downloads", "crisismmd_v2.0"}
    if not os.path.isdir(RAW_DIR):
        print(f"[ERROR] raw_dataset directory not found: {RAW_DIR}")
        return categories

    try:
        for name in sorted(os.listdir(RAW_DIR)):
            full = npath(RAW_DIR, name)
            if not os.path.isdir(full):
                continue
            if name.startswith("_") or name.lower() in skip_dirs:
                continue
            categories.append(name)
    except OSError as exc:
        print(f"[ERROR] Could not list raw categories in '{RAW_DIR}': {exc}")
    return categories


def clean_and_resize() -> Dict[str, Dict[str, int]]:
    print("=" * 60)
    print("STEP 2 - PHASE 1: CLEAN + RESIZE")
    print("=" * 60)

    safe_remove_dir(CLEAN_DIR)
    try:
        os.makedirs(CLEAN_DIR, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Could not create clean_dataset '{CLEAN_DIR}': {exc}")
        return {}

    seen_hashes = set()
    stats: Dict[str, Dict[str, int]] = {}

    categories = _list_raw_categories()
    if "_downloads" in categories:
        categories.remove("_downloads")

    for category in categories:
        raw_cat_dir = npath(RAW_DIR, category)
        clean_cat_dir = npath(CLEAN_DIR, category)

        try:
            os.makedirs(clean_cat_dir, exist_ok=True)
        except OSError as exc:
            print(f"[ERROR] Could not create clean category '{clean_cat_dir}': {exc}")
            continue

        saved = 0
        dupes = 0
        corrupt = 0
        read_errors = 0

        try:
            files = [f for f in os.listdir(raw_cat_dir) if f.lower().endswith(IMG_EXTS)]
        except OSError as exc:
            print(f"[ERROR] Could not list files in '{raw_cat_dir}': {exc}")
            stats[category] = {"saved": 0, "duplicates_removed": 0, "errors": 1}
            continue

        for fname in tqdm(files, desc=f"  {category}"):
            src = npath(raw_cat_dir, fname)
            src_hash = get_md5(src)
            if not src_hash:
                read_errors += 1
                continue
            if src_hash in seen_hashes:
                dupes += 1
                continue
            seen_hashes.add(src_hash)

            stem = os.path.splitext(fname)[0]
            dst = npath(clean_cat_dir, f"{stem}.jpg")
            try:
                with Image.open(src) as img:
                    img = img.convert("RGB")
                    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                    img.save(dst, "JPEG", quality=90)
                saved += 1
            except UnidentifiedImageError as exc:
                print(f"[WARN] Corrupt image skipped '{src}': {exc}")
                corrupt += 1
            except OSError as exc:
                print(f"[WARN] Failed processing '{src}': {exc}")
                read_errors += 1

        stats[category] = {
            "saved": saved,
            "duplicates_removed": dupes,
            "errors": corrupt + read_errors,
        }
        print(f"  {category:14s}: saved={saved}, duplicates={dupes}, errors={corrupt + read_errors}")

    return stats


def split_dataset() -> Dict[str, Dict[str, int]]:
    print("\n" + "=" * 60)
    print("STEP 2 - PHASE 2: TRAIN/TEST SPLIT (70/30)")
    print("=" * 60)

    safe_remove_dir(SPLIT_DIR)
    try:
        os.makedirs(SPLIT_DIR, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Could not create split_dataset '{SPLIT_DIR}': {exc}")
        return {}

    split_stats: Dict[str, Dict[str, int]] = {}
    if not os.path.isdir(CLEAN_DIR):
        print(f"[ERROR] clean_dataset directory not found: {CLEAN_DIR}")
        return split_stats

    try:
        categories = sorted([c for c in os.listdir(CLEAN_DIR) if os.path.isdir(npath(CLEAN_DIR, c))])
    except OSError as exc:
        print(f"[ERROR] Could not list clean categories: {exc}")
        return split_stats

    for category in categories:
        cat_dir = npath(CLEAN_DIR, category)
        try:
            files = [f for f in os.listdir(cat_dir) if f.lower().endswith(".jpg")]
        except OSError as exc:
            print(f"[ERROR] Could not list '{cat_dir}': {exc}")
            continue

        if not files:
            print(f"[WARN] Empty class skipped in split: {category}")
            continue

        if len(files) < 2:
            train_files = files
            test_files: List[str] = []
        else:
            train_files, test_files = train_test_split(files, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        for split_name, file_list in (("train", train_files), ("test", test_files)):
            out_dir = npath(SPLIT_DIR, split_name, category)
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as exc:
                print(f"[ERROR] Could not create split directory '{out_dir}': {exc}")
                continue

            for fname in file_list:
                src = npath(cat_dir, fname)
                dst = npath(out_dir, fname)
                try:
                    shutil.copy2(src, dst)
                except (OSError, shutil.Error) as exc:
                    print(f"[WARN] Copy failed '{src}' -> '{dst}': {exc}")

        split_stats[category] = {"train": len(train_files), "test": len(test_files)}
        print(f"  {category:14s}: train={len(train_files)}, test={len(test_files)}")

    return split_stats


def print_summary(clean_stats: Dict[str, Dict[str, int]], split_stats: Dict[str, Dict[str, int]]) -> None:
    print("\n" + "=" * 60)
    print("STEP 2 SUMMARY")
    print("=" * 60)

    total_clean = sum(v.get("saved", 0) for v in clean_stats.values())
    total_train = sum(v.get("train", 0) for v in split_stats.values())
    total_test = sum(v.get("test", 0) for v in split_stats.values())

    print(f"  cleaned images: {total_clean}")
    print(f"  train images  : {total_train}")
    print(f"  test images   : {total_test}")
    print(f"  total split   : {total_train + total_test}")
    print(f"  target size   : {IMG_SIZE[0]}x{IMG_SIZE[1]}")


def main() -> int:
    clean_stats = clean_and_resize()
    split_stats = split_dataset()
    print_summary(clean_stats, split_stats)
    if not split_stats:
        print("[ERROR] Split statistics are empty. Check step1 output and rerun.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
