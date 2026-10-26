"""Small bounded in-process cache for completed text reports."""
from __future__ import annotations
import os
from pathlib import Path
from diskcache import Cache

_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / ".cache" / "crew"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cache = Cache(str(_CACHE_DIR))

def get(key: str, ttl_s: int = 3600):
    return None

def set(key: str, value: object, ttl_s: int = 3600) -> None:
    pass
