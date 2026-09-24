from __future__ import annotations
import io
import ipaddress
import socket
from urllib.parse import urlparse
import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError
from .prep import resize_rgb_to_max_edge

MAX_BYTES = 12 * 1024 * 1024

def _safe_host(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
        return all(not ipaddress.ip_address(item[4][0]).is_private and not ipaddress.ip_address(item[4][0]).is_loopback for item in addresses)
    except OSError:
        return False

def safe_fetch_image(url: str, timeout_s: float = 10) -> np.ndarray | None:
    if not _safe_host(url):
        return None
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"User-Agent": "DisasterRes-Net/1.0"}) as client:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if len(response.content) > MAX_BYTES or (content_type and not content_type.startswith("image/")):
                return None
            with Image.open(io.BytesIO(response.content)) as opened:
                opened.verify()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return resize_rgb_to_max_edge(np.asarray(image, dtype=np.uint8))
    except (httpx.HTTPError, OSError, UnidentifiedImageError, ValueError):
        return None
