"""Step 2: clean, deduplicate (MD5 + perceptual dHash), filter banners, resize, and split dataset."""

import concurrent.futures
import hashlib
import json
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
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


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute 64-bit perceptual difference hash (dHash) to catch near-duplicate images."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def _is_degenerate_banner(img: Image.Image) -> bool:
    """Return True if image is too small (<32x32) or a blank solid-color graphic."""
    w, h = img.size
    if w < 32 or h < 32:
        return True
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return float(np.std(gray)) < 2.0


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


def _process_single_image(src: str, dst: str) -> Tuple[str, str, str]:
    """Process single image: compute MD5 + perceptual dHash, check banner quality, and convert/resize."""
    try:
        src_hash = get_md5(src)
        if not src_hash:
            return "", "", "error"

        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            with Image.open(dst) as existing_img:
                dhash = compute_dhash(existing_img)
            return src_hash, dhash, "saved"

        with Image.open(src) as img:
            img = img.convert("RGB")
            if _is_degenerate_banner(img):
                return src_hash, "", "corrupt"
            dhash = compute_dhash(img)
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            img.save(dst, "JPEG", quality=90)
        return src_hash, dhash, "saved"
    except UnidentifiedImageError:
        return "", "", "corrupt"
    except Exception:
        return "", "", "error"


def clean_and_resize() -> Dict[str, Dict[str, int]]:
    print("=" * 60)
    print("STEP 2 - PHASE 1: CLEAN + DEDUPLICATE (MD5 + dHash) + RESIZE")
    print("=" * 60)

    try:
        os.makedirs(CLEAN_DIR, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Could not create clean_dataset '{CLEAN_DIR}': {exc}")
        return {}

    seen_hashes = set()
    seen_dhashes = set()
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

        try:
            files = [f for f in os.listdir(raw_cat_dir) if f.lower().endswith(IMG_EXTS)]
        except OSError as exc:
            print(f"[ERROR] Could not list files in '{raw_cat_dir}': {exc}")
            stats[category] = {"saved": 0, "duplicates_removed": 0, "perceptual_duplicates_removed": 0, "errors": 1}
            continue

        saved = 0
        dupes = 0
        perceptual_dupes = 0
        corrupt = 0
        read_errors = 0

        tasks = []
        for fname in files:
            src = npath(raw_cat_dir, fname)
            stem = os.path.splitext(fname)[0]
            dst = npath(clean_cat_dir, f"{stem}.jpg")
            tasks.append((src, dst))

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_dst = {executor.submit(_process_single_image, src, dst): dst for src, dst in tasks}
            for future in tqdm(
                concurrent.futures.as_completed(future_to_dst),
                total=len(future_to_dst),
                desc=f"  {category:14s}",
            ):
                src_hash, dhash, status = future.result()
                if status == "saved":
                    if src_hash in seen_hashes or (dhash and dhash in seen_dhashes):
                        if src_hash in seen_hashes:
                            dupes += 1
                        else:
                            perceptual_dupes += 1
                        dst_file = future_to_dst[future]
                        if os.path.isfile(dst_file):
                            try:
                                os.remove(dst_file)
                            except OSError:
                                pass
                    else:
                        seen_hashes.add(src_hash)
                        if dhash:
                            seen_dhashes.add(dhash)
                        saved += 1
                elif status == "corrupt":
                    corrupt += 1
                else:
                    read_errors += 1

        stats[category] = {
            "saved": saved,
            "duplicates_removed": dupes + perceptual_dupes,
            "exact_md5_duplicates": dupes,
            "perceptual_dhash_duplicates": perceptual_dupes,
            "errors": corrupt + read_errors,
        }
        print(
            f"  {category:14s}: saved={saved}, md5_dupes={dupes}, "
            f"dhash_dupes={perceptual_dupes}, errors={corrupt + read_errors}"
        )

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
            files = sorted([f for f in os.listdir(cat_dir) if f.lower().endswith(".jpg")])
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

        def _copy_file(s: str, d: str) -> None:
            try:
                if os.path.isfile(d) and os.path.getsize(d) > 0:
                    return
                shutil.copy2(s, d)
            except (OSError, shutil.Error):
                pass

        copy_tasks = []
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
                copy_tasks.append((src, dst))

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(lambda pair: _copy_file(pair[0], pair[1]), copy_tasks))

        split_stats[category] = {"train": len(train_files), "test": len(test_files)}
        print(f"  {category:14s}: train={len(train_files)}, test={len(test_files)}")

    return split_stats


def print_summary(clean_stats: Dict[str, Dict[str, int]], split_stats: Dict[str, Dict[str, int]]) -> None:
    print("\n" + "=" * 60)
    print("STEP 2 SUMMARY")
    print("=" * 60)

    total_clean = sum(v.get("saved", 0) for v in clean_stats.values())
    total_dupes = sum(v.get("duplicates_removed", 0) for v in clean_stats.values())
    total_train = sum(v.get("train", 0) for v in split_stats.values())
    total_test = sum(v.get("test", 0) for v in split_stats.values())

    print(f"  cleaned images   : {total_clean}")
    print(f"  duplicates pruned: {total_dupes}")
    print(f"  train images     : {total_train}")
    print(f"  test images      : {total_test}")
    print(f"  total split      : {total_train + total_test}")
    print(f"  target size      : {IMG_SIZE[0]}x{IMG_SIZE[1]}")


def main() -> int:
    clean_stats = clean_and_resize()
    split_stats = split_dataset()
    print_summary(clean_stats, split_stats)
    if not split_stats:
        print("[ERROR] Split statistics are empty. Check step1 output and rerun.")
        return 1

    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("DisasterRes-Net")

        with mlflow.start_run(run_name="step2_preprocessing"):
            mlflow.set_tags({"step": "step2_preprocessing", "framework": "DisasterRes-Net"})
            mlflow.log_params(
                {
                    "img_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
                    "test_size": TEST_SIZE,
                    "random_seed": RANDOM_SEED,
                    "deduplication": "md5+dhash64",
                }
            )
            total_clean = sum(v.get("saved", 0) for v in clean_stats.values())
            total_dupes = sum(v.get("duplicates_removed", 0) for v in clean_stats.values())
            total_train = sum(v.get("train", 0) for v in split_stats.values())
            total_test = sum(v.get("test", 0) for v in split_stats.values())

            metrics_dict = {
                "clean_images_total": float(total_clean),
                "duplicates_removed_total": float(total_dupes),
                "train_split_total": float(total_train),
                "test_split_total": float(total_test),
            }
            for cat, c_stat in clean_stats.items():
                metrics_dict[f"clean_saved_{cat}"] = float(c_stat.get("saved", 0))
                metrics_dict[f"duplicates_{cat}"] = float(c_stat.get("duplicates_removed", 0))
            for cat, s_stat in split_stats.items():
                metrics_dict[f"train_{cat}"] = float(s_stat.get("train", 0))
                metrics_dict[f"test_{cat}"] = float(s_stat.get("test", 0))

            mlflow.log_metrics(metrics_dict)

            summary_path = npath(CLEAN_DIR, "step2_summary.json")
            os.makedirs(CLEAN_DIR, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as fobj:
                json.dump(
                    {
                        "clean_stats": clean_stats,
                        "split_stats": split_stats,
                        "total_clean": total_clean,
                        "total_train": total_train,
                        "total_test": total_test,
                    },
                    fobj,
                    indent=2,
                )
            mlflow.log_artifact(summary_path)
            print("[INFO] Step 2 preprocessing successfully logged to MLflow experiment 'DisasterRes-Net'.")
    except Exception as exc:
        print(f"[WARN] Could not log Step 2 run to MLflow: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
