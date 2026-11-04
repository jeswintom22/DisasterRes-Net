"""Small bounded in-process cache for completed text reports and resolution metadata."""
from __future__ import annotations
import os
from pathlib import Path
from diskcache import Cache

_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / ".cache" / "crew"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cache = Cache(str(_CACHE_DIR))


def get(key: str, ttl_s: int = 3600):
    if os.getenv("DISASTERRES_DISABLE_CACHE") == "1":
        return None
    # Avoid returning stale full-crew report payloads across mocked pytest runs
    if key.startswith("v3:") and os.getenv("PYTEST_CURRENT_TEST"):
        return None
    return _cache.get(key, default=None)


def set(key: str, value: object, ttl_s: int = 3600) -> None:
    if os.getenv("DISASTERRES_DISABLE_CACHE") == "1":
        return
    if key.startswith("v3:") and os.getenv("PYTEST_CURRENT_TEST"):
        return
    _cache.set(key, value, expire=ttl_s)
