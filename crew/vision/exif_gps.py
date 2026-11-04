"""EXIF GPS coordinate extraction for uploaded field imagery (`FR-7`, `EC-5`, `EC-6`)."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any
from PIL import Image

_GPS_INFO_TAG = 0x8825  # 34853


def _to_float(val: Any) -> float:
    if isinstance(val, tuple) and len(val) == 2:
        return float(val[0]) / float(val[1]) if val[1] else 0.0
    if hasattr(val, "numerator") and hasattr(val, "denominator"):
        return float(val.numerator) / float(val.denominator) if val.denominator else 0.0
    return float(val)


def _dms_to_decimal(dms: Any, ref: str) -> float | None:
    if not isinstance(dms, (tuple, list)) or len(dms) < 3:
        return None
    try:
        deg = _to_float(dms[0])
        minutes = _to_float(dms[1])
        seconds = _to_float(dms[2])
        decimal = deg + (minutes / 60.0) + (seconds / 3600.0)
        if str(ref or "").strip().upper() in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _open_image_source(image_source: str | bytes | Path) -> Image.Image | None:
    try:
        if isinstance(image_source, Path):
            return Image.open(image_source)
        if isinstance(image_source, bytes):
            return Image.open(io.BytesIO(image_source))
        if isinstance(image_source, str):
            stripped = image_source.strip()
            if not stripped:
                return None
            if stripped.startswith("data:") and "," in stripped:
                b64_part = stripped.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_part)
                return Image.open(io.BytesIO(raw_bytes))
            # Check if it is an existing file path first
            try:
                p = Path(stripped)
                if len(stripped) < 260 and p.exists() and p.is_file():
                    return Image.open(p)
            except OSError:
                pass
            # Otherwise treat as base64 string
            raw_bytes = base64.b64decode(stripped)
            return Image.open(io.BytesIO(raw_bytes))
    except Exception:
        return None
    return None


def extract_exif_gps(image_source: str | bytes | Path) -> tuple[float, float] | None:
    """
    Extract WGS84 (latitude, longitude) decimal degrees from an image's EXIF GPSInfo IFD.
    Returns None if EXIF GPS tags are absent, corrupted, out of bounds, or Null Island (0.0, 0.0).
    """
    img = _open_image_source(image_source)
    if img is None:
        return None

    try:
        gps_ifd: dict[int, Any] | None = None
        exif = img.getexif()
        if exif:
            try:
                gps_ifd = exif.get_ifd(_GPS_INFO_TAG)
            except Exception:
                gps_ifd = None
            if not gps_ifd and _GPS_INFO_TAG in exif:
                cand = exif.get(_GPS_INFO_TAG)
                if isinstance(cand, dict):
                    gps_ifd = cand

        if not gps_ifd and hasattr(img, "_getexif"):
            raw_exif = img._getexif()  # type: ignore[attr-defined]
            if isinstance(raw_exif, dict):
                cand = raw_exif.get(_GPS_INFO_TAG)
                if isinstance(cand, dict):
                    gps_ifd = cand

        if not gps_ifd:
            return None

        # Standard EXIF GPS tags:
        # 1: GPSLatitudeRef ('N'/'S'), 2: GPSLatitude, 3: GPSLongitudeRef ('E'/'W'), 4: GPSLongitude
        lat_ref = str(gps_ifd.get(1, "N"))
        lat_dms = gps_ifd.get(2)
        lon_ref = str(gps_ifd.get(3, "E"))
        lon_dms = gps_ifd.get(4)

        if lat_dms is None or lon_dms is None:
            return None

        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)
        if lat is None or lon is None:
            return None

        # EC-6: Reject Null Island (0.0, 0.0) and out-of-range coordinates
        if abs(lat) < 1e-4 and abs(lon) < 1e-4:
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None

        return (lat, lon)
    except Exception:
        return None
    finally:
        try:
            img.close()
        except Exception:
            pass

