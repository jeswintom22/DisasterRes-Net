"""Small bounded in-process cache for completed text reports."""
from __future__ import annotations
import os
from pathlib import Path
from diskcache import Cache

_CACHE_DIR = Path(".cache/crew")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cache = Cache(str(_CACHE_DIR))

def get(key: str, ttl_s: int = 3600):
    return _cache.get(key)

def set(key: str, value: object, ttl_s: int = 3600) -> None:
    _cache.set(key, value, expire=ttl_s)
