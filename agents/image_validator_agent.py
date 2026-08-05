"""Validate Twitter image candidates before saving them to raw_dataset."""

import csv
import os
from datetime import datetime
from io import BytesIO
from typing import Iterable, List, Optional, Set

import requests
from PIL import Image, UnidentifiedImageError

from agents.schemas import ImageCandidate
from config.agent_config import PROVENANCE_PATH, RAW_DATASET_DIR

try:
    import imagehash
except ImportError:
    imagehash = None


MAX_BYTES = 15 * 1024 * 1024
MIN_BYTES = 20 * 1024
MIN_EDGE = 200
PROVENANCE_COLUMNS = [
    "filename",
    "disaster_class",
    "source_tweet_id",
    "event_id",
    "event_source",
    "tweet_created_at",
    "collected_at",
    "latitude",
    "longitude",
    "place_name",
]


class ImageValidatorAgent:
    def __init__(self, raw_dataset_dir: str = RAW_DATASET_DIR, provenance_path: str = PROVENANCE_PATH):
        self.raw_dataset_dir = raw_dataset_dir
        self.provenance_path = provenance_path
        os.makedirs(raw_dataset_dir, exist_ok=True)
        self._ensure_provenance_file()
        self.seen_hashes: Optional[Set[str]] = None

    def process(self, candidates: Iterable[ImageCandidate]) -> int:
        candidate_list: List[ImageCandidate] = list(candidates)
        if not candidate_list:
            return 0
        if self.seen_hashes is None:
            self.seen_hashes = self._load_existing_hashes()
        saved = 0
        for candidate in candidate_list:
            if self._process_one(candidate):
                saved += 1
        return saved

    def _process_one(self, candidate: ImageCandidate) -> bool:
        try:
            content = self._download(candidate.url)
            if len(content) < MIN_BYTES:
                return False
            with Image.open(BytesIO(content)) as image:
                image.verify()
            with Image.open(BytesIO(content)) as image:
                image = image.convert("RGB")
                width, height = image.size
                if width < MIN_EDGE or height < MIN_EDGE:
                    return False
                phash = _perceptual_hash(image)
                if phash is None:
                    print("[OBSERVATION] imagehash is not installed; cannot validate candidate image hashes.")
                    return False
                if self.seen_hashes is None:
                    self.seen_hashes = set()
                if phash in self.seen_hashes:
                    return False
                self.seen_hashes.add(phash)
                filename = self._filename(candidate)
                class_dir = os.path.join(self.raw_dataset_dir, candidate.disaster_class)
                os.makedirs(class_dir, exist_ok=True)
                image.save(os.path.join(class_dir, filename), format="JPEG", quality=92)
                self._append_provenance(filename, candidate)
                return True
        except (requests.RequestException, OSError, UnidentifiedImageError, ValueError):
            return False

    def _download(self, url: str) -> bytes:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError("image exceeds 15MB size cap")
            chunks.append(chunk)
        return b"".join(chunks)

    def _load_existing_hashes(self) -> Set[str]:
        hashes: Set[str] = set()
        for root, _, names in os.walk(self.raw_dataset_dir):
            for name in names:
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                path = os.path.join(root, name)
                try:
                    with Image.open(path) as image:
                        phash = _perceptual_hash(image.convert("RGB"))
                        if phash is not None:
                            hashes.add(phash)
                except (OSError, UnidentifiedImageError, ValueError):
                    continue
        return hashes

    def _ensure_provenance_file(self) -> None:
        parent = os.path.dirname(self.provenance_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(self.provenance_path):
            return
        with open(self.provenance_path, "w", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=PROVENANCE_COLUMNS)
            writer.writeheader()

    def _append_provenance(self, filename: str, candidate: ImageCandidate) -> None:
        with open(self.provenance_path, "a", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=PROVENANCE_COLUMNS)
            writer.writerow(
                {
                    "filename": filename,
                    "disaster_class": candidate.disaster_class,
                    "source_tweet_id": candidate.source_tweet_id,
                    "event_id": candidate.event_id,
                    "event_source": candidate.event_source,
                    "tweet_created_at": candidate.tweet_created_at.isoformat(),
                    "collected_at": datetime.utcnow().isoformat(),
                    "latitude": candidate.latitude,
                    "longitude": candidate.longitude,
                    "place_name": candidate.place_name or "",
                }
            )

    def _filename(self, candidate: ImageCandidate) -> str:
        event_id = _safe_name(candidate.event_id)
        tweet_id = _safe_name(candidate.source_tweet_id)
        return f"twitter_{event_id}_{tweet_id}.jpg"


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return cleaned[:80] or "unknown"


def _perceptual_hash(image: Image.Image) -> Optional[str]:
    if imagehash is None:
        return None
    return str(imagehash.phash(image))
