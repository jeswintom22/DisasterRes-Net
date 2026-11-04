from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

from .ai_resolver import ALLOWED_HAZARDS, extract_query_entities

# Deterministic alias & hierarchy table for common disaster queries and transliterations.
LOCATION_ALIASES: dict[str, str] = {
    "wayand": "Wayanad, Kerala, India",
    "wayanad": "Wayanad, Kerala, India",
    "wayanadu": "Wayanad, Kerala, India",
    "chooralmala": "Chooralmala, Wayanad, Kerala, India",
    "mundakkai": "Mundakkai, Wayanad, Kerala, India",
    "chamoli": "Chamoli, Uttarakhand, India",
    "joshimath": "Joshimath, Uttarakhand, India",
    "valencia": "Valencia, Valencian Community, Spain",
    "paiporta": "Paiporta, Valencia, Spain",
    "antakya": "Antakya, Hatay, Türkiye",
    "hatay": "Hatay, Türkiye",
    "kahramanmaras": "Kahramanmaraş, Türkiye",
    "turkiye": "Türkiye",
    "turkey": "Türkiye",
    "derna": "Derna, Libya",
    "al haouz": "Al Haouz, Marrakesh-Safi, Morocco",
    "marrakesh": "Marrakesh, Marrakesh-Safi, Morocco",
    "bhola": "Bhola, Barisal, Bangladesh",
    "sylhet": "Sylhet, Bangladesh",
    "noto": "Noto, Ishikawa, Japan",
    "ishikawa": "Ishikawa, Japan",
    "lahaina": "Lahaina, Maui, Hawaii, United States",
    "maui": "Maui, Hawaii, United States",
    "acapulco": "Acapulco, Guerrero, Mexico",
    "beira": "Beira, Sofala, Mozambique",
}

_STRUCTURED_GAZETTEER: dict[str, dict[str, str]] = {
    "wayanad, kerala, india": {
        "locality": "Wayanad",
        "admin_area": "Kerala",
        "country": "India",
        "country_iso2": "in",
    },
    "chooralmala, wayanad, kerala, india": {
        "locality": "Chooralmala",
        "admin_area": "Kerala",
        "country": "India",
        "country_iso2": "in",
    },
    "mundakkai, wayanad, kerala, india": {
        "locality": "Mundakkai",
        "admin_area": "Kerala",
        "country": "India",
        "country_iso2": "in",
    },
    "chamoli, uttarakhand, india": {
        "locality": "Chamoli",
        "admin_area": "Uttarakhand",
        "country": "India",
        "country_iso2": "in",
    },
    "joshimath, uttarakhand, india": {
        "locality": "Joshimath",
        "admin_area": "Uttarakhand",
        "country": "India",
        "country_iso2": "in",
    },
    "valencia, valencian community, spain": {
        "locality": "Valencia",
        "admin_area": "Valencian Community",
        "country": "Spain",
        "country_iso2": "es",
    },
    "paiporta, valencia, spain": {
        "locality": "Paiporta",
        "admin_area": "Valencia",
        "country": "Spain",
        "country_iso2": "es",
    },
    "antakya, hatay, türkiye": {
        "locality": "Antakya",
        "admin_area": "Hatay",
        "country": "Türkiye",
        "country_iso2": "tr",
    },
    "hatay, türkiye": {
        "locality": "Hatay",
        "admin_area": "Hatay",
        "country": "Türkiye",
        "country_iso2": "tr",
    },
    "kahramanmaraş, türkiye": {
        "locality": "Kahramanmaraş",
        "admin_area": "Kahramanmaraş",
        "country": "Türkiye",
        "country_iso2": "tr",
    },
    "derna, libya": {
        "locality": "Derna",
        "admin_area": "Derna District",
        "country": "Libya",
        "country_iso2": "ly",
    },
    "al haouz, marrakesh-safi, morocco": {
        "locality": "Al Haouz",
        "admin_area": "Marrakesh-Safi",
        "country": "Morocco",
        "country_iso2": "ma",
    },
    "marrakesh, marrakesh-safi, morocco": {
        "locality": "Marrakesh",
        "admin_area": "Marrakesh-Safi",
        "country": "Morocco",
        "country_iso2": "ma",
    },
    "bhola, barisal, bangladesh": {
        "locality": "Bhola",
        "admin_area": "Barisal",
        "country": "Bangladesh",
        "country_iso2": "bd",
    },
    "sylhet, bangladesh": {
        "locality": "Sylhet",
        "admin_area": "Sylhet",
        "country": "Bangladesh",
        "country_iso2": "bd",
    },
    "noto, ishikawa, japan": {
        "locality": "Noto",
        "admin_area": "Ishikawa",
        "country": "Japan",
        "country_iso2": "jp",
    },
    "lahaina, maui, hawaii, united states": {
        "locality": "Lahaina",
        "admin_area": "Hawaii",
        "country": "United States",
        "country_iso2": "us",
    },
    "acapulco, guerrero, mexico": {
        "locality": "Acapulco",
        "admin_area": "Guerrero",
        "country": "Mexico",
        "country_iso2": "mx",
    },
    "beira, sofala, mozambique": {
        "locality": "Beira",
        "admin_area": "Sofala",
        "country": "Mozambique",
        "country_iso2": "mz",
    },
}

_COUNTRY_ISO2_MAP: dict[str, tuple[str, str]] = {
    "india": ("India", "in"),
    "spain": ("Spain", "es"),
    "türkiye": ("Türkiye", "tr"),
    "turkiye": ("Türkiye", "tr"),
    "turkey": ("Türkiye", "tr"),
    "libya": ("Libya", "ly"),
    "lebanon": ("Lebanon", "lb"),
    "morocco": ("Morocco", "ma"),
    "bangladesh": ("Bangladesh", "bd"),
    "japan": ("Japan", "jp"),
    "united states": ("United States", "us"),
    "usa": ("United States", "us"),
    "mexico": ("Mexico", "mx"),
    "mozambique": ("Mozambique", "mz"),
    "chile": ("Chile", "cl"),
    "philippines": ("Philippines", "ph"),
    "venezuela": ("Venezuela", "ve"),
    "nepal": ("Nepal", "np"),
    "pakistan": ("Pakistan", "pk"),
    "indonesia": ("Indonesia", "id"),
    "italy": ("Italy", "it"),
    "germany": ("Germany", "de"),
    "brazil": ("Brazil", "br"),
    "myanmar": ("Myanmar", "mm"),
}

_ACTION_AND_STOP_WORDS = {
    "assess", "assessment", "analyze", "analyse", "analysis", "check",
    "evaluate", "evaluation", "explain", "show", "report", "find",
    "get", "summarize", "summary", "tell", "me", "about", "the", "a", "an",
    "current", "latest", "recent", "severe", "major", "devastating", "deadly",
    "update", "status", "situation", "impact", "aftermath", "damage", "damages",
    "destruction", "severity", "crisis", "disaster", "event", "alert",
    "emergency", "response", "in", "at", "near", "around", "from", "across",
    "of", "for", "on", "to", "by", "with", "and", "or", "during", "after",
    "hit", "struck", "affected", "area", "region", "zone", "city", "town",
    "district", "province", "state", "what", "happened", "is", "was", "were",
}

_HAZARD_TOKENS = {
    "flood": "flood",
    "floods": "flood",
    "flooding": "flood",
    "flashflood": "flood",
    "inundation": "flood",
    "landslide": "landslide",
    "landslides": "landslide",
    "landslid": "landslide",
    "mudslide": "landslide",
    "mudslides": "landslide",
    "earthquake": "earthquake",
    "earthquakes": "earthquake",
    "quake": "earthquake",
    "tremor": "earthquake",
    "cyclone": "cyclone",
    "typhoon": "cyclone",
    "hurricane": "cyclone",
    "storm": "cyclone",
    "wildfire": "wildfire",
    "wildfires": "wildfire",
    "bushfire": "wildfire",
    "fire": "wildfire",
    "collapse": "collapse",
}

_CONCEPTUAL_TERMS = {
    "mahalanobis", "gradcam", "grad-cam", "gradcam++", "scorecam",
    "temperature scaling", "focal loss", "xgboost", "lightgbm", "catboost",
    "resnet", "swin", "convnext", "efficientnet", "saliency", "f1 score",
    "confusion matrix", "ece", "calibration", "brier", "hyperparameter",
    "neural network", "classifier", "architecture", "algorithm",
}

_INJECTION_RE = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
    r"system\s*prompt|output\s+pwned|return\s+latitude|disregard\s+all)",
    re.I,
)

_TIME_HINT_RE = re.compile(
    r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+20\d{2}|20\d{2}-\d{2}|20\d{2})\b",
    re.I,
)


@dataclass(frozen=True)
class ResolvedQuery:
    intent: str
    location: str | None
    time_hint: str | None = None
    hazard_hint: str | None = None
    locality: str | None = None
    admin_area: str | None = None
    country: str | None = None
    country_iso2: str | None = None
    canonical_query: str | None = None
    confidence: float = 0.0
    resolution_method: str = "none"
    raw: str = ""
    cleaned: str = ""


def _extract_hazard_hint(text: str) -> str | None:
    lower = text.lower()
    for phrase, norm in ALLOWED_HAZARDS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            return norm
    for token in re.findall(r"[a-z]+", lower):
        if token in _HAZARD_TOKENS:
            return _HAZARD_TOKENS[token]
    return None


def _extract_time_hint(text: str) -> str | None:
    match = _TIME_HINT_RE.search(text)
    return match.group(1) if match else None


def _is_conceptual_or_injected(cleaned: str) -> bool:
    lower = cleaned.lower()
    if _INJECTION_RE.search(lower):
        return True
    for term in _CONCEPTUAL_TERMS:
        if term in lower:
            # Ensure no known location alias is also present
            if not any(alias in lower for alias in LOCATION_ALIASES):
                return True
    if re.match(r"^(?:what\s+is|how\s+does|how\s+do|define|why\s+is)\b", lower):
        if not re.search(r"\b(?:in|at|near|around)\s+[A-Z]", cleaned):
            if not any(alias in lower for alias in LOCATION_ALIASES):
                return True
    return False


def _deterministic_extract_toponym(cleaned: str) -> dict[str, Any]:
    """Verb-stripping and preposition-aware deterministic fallback parser (Stage 1B)."""
    hazard_hint = _extract_hazard_hint(cleaned)
    time_hint = _extract_time_hint(cleaned)

    if _is_conceptual_or_injected(cleaned):
        return {
            "locality": None,
            "admin_area": None,
            "country": None,
            "country_iso2": None,
            "canonical_query": None,
            "hazard_hint": hazard_hint,
            "time_hint": time_hint,
            "confidence": 0.0,
            "resolution_method": "none",
        }

    # Remove time_hint tokens before extracting location
    working = _TIME_HINT_RE.sub(" ", cleaned)

    # 1. Check preposition pattern first: "in/at/near/around <Place>"
    prep_match = re.search(
        r"\b(?:at|in|near|around|across|from)\s+([\w .,'\-]+?)(?:\s+(?:in|during|after|from|on|where|with)\b|[?.!,]|$)",
        working,
        re.I,
    )
    candidate_tokens: list[str] = []
    if prep_match:
        raw_seg = prep_match.group(1).strip(" .,?!")
        for tok in raw_seg.split():
            clean_tok = tok.strip(" .,?!")
            if clean_tok.lower() not in _ACTION_AND_STOP_WORDS and clean_tok.lower() not in _HAZARD_TOKENS:
                candidate_tokens.append(clean_tok)

    # 2. If no preposition match (e.g. "Assess Valencia flood damage", "Wayanad landslide"),
    # strip action verbs, stop words, and hazard nouns from the full query.
    if not candidate_tokens:
        for tok in re.split(r"\s+", working):
            clean_tok = tok.strip(" .,?!:;\"'()")
            if not clean_tok:
                continue
            low = clean_tok.lower()
            if low in _ACTION_AND_STOP_WORDS or low in _HAZARD_TOKENS:
                continue
            if re.match(r"^\d+$", low):
                continue
            candidate_tokens.append(clean_tok)

    if not candidate_tokens:
        return {
            "locality": None,
            "admin_area": None,
            "country": None,
            "country_iso2": None,
            "canonical_query": None,
            "hazard_hint": hazard_hint,
            "time_hint": time_hint,
            "confidence": 0.0,
            "resolution_method": "none",
        }

    raw_place = " ".join(candidate_tokens).strip(" ,")
    raw_lower = raw_place.lower()

    # Check if full phrase or individual tokens match LOCATION_ALIASES
    canonical = LOCATION_ALIASES.get(raw_lower)
    if not canonical:
        for tok in candidate_tokens:
            alias_hit = LOCATION_ALIASES.get(tok.lower().strip(","))
            if alias_hit:
                canonical = alias_hit
                break

    # Check if a country token was also mentioned alongside a city (e.g., "Türkiye Antakya" or "Tripoli Lebanon")
    detected_country: str | None = None
    detected_iso2: str | None = None
    remaining_tokens: list[str] = []
    for tok in candidate_tokens:
        low_tok = tok.lower().strip(",")
        if low_tok in _COUNTRY_ISO2_MAP:
            detected_country, detected_iso2 = _COUNTRY_ISO2_MAP[low_tok]
        else:
            remaining_tokens.append(tok.strip(","))

    if canonical:
        gaz = _STRUCTURED_GAZETTEER.get(canonical.lower(), {})
        return {
            "locality": gaz.get("locality", canonical.split(",")[0].strip()),
            "admin_area": gaz.get("admin_area"),
            "country": gaz.get("country", detected_country),
            "country_iso2": gaz.get("country_iso2", detected_iso2),
            "canonical_query": canonical,
            "hazard_hint": hazard_hint,
            "time_hint": time_hint,
            "confidence": 0.95,
            "resolution_method": "regex_fallback",
        }

    if remaining_tokens:
        locality_str = " ".join(remaining_tokens).strip(" ,")
        if detected_country and detected_country.lower() not in locality_str.lower():
            canonical_query = f"{locality_str}, {detected_country}"
        else:
            canonical_query = locality_str
    elif detected_country:
        locality_str = detected_country
        canonical_query = detected_country
    else:
        locality_str = raw_place
        canonical_query = raw_place

    return {
        "locality": locality_str,
        "admin_area": None,
        "country": detected_country,
        "country_iso2": detected_iso2,
        "canonical_query": canonical_query,
        "hazard_hint": hazard_hint,
        "time_hint": time_hint,
        "confidence": 0.75,
        "resolution_method": "regex_fallback",
    }


def _enrich_with_gazetteer(extracted: dict[str, Any]) -> dict[str, Any]:
    """Ensure known aliases (like Wayanad -> Wayanad, Kerala, India) always normalize deterministically."""
    loc_cand = (extracted.get("locality") or extracted.get("canonical_query") or "").strip()
    if not loc_cand:
        return extracted

    alias_target = LOCATION_ALIASES.get(loc_cand.lower())
    if not alias_target:
        first_part = loc_cand.split(",")[0].strip().lower()
        alias_target = LOCATION_ALIASES.get(first_part)

    if alias_target:
        gaz = _STRUCTURED_GAZETTEER.get(alias_target.lower(), {})
        extracted = dict(extracted)
        extracted["canonical_query"] = alias_target
        extracted["locality"] = gaz.get("locality") or extracted.get("locality")
        extracted["admin_area"] = gaz.get("admin_area") or extracted.get("admin_area")
        extracted["country"] = gaz.get("country") or extracted.get("country")
        extracted["country_iso2"] = gaz.get("country_iso2") or extracted.get("country_iso2")
    return extracted


def resolve_query(
    query: str,
    *,
    offline: bool | None = None,
    use_ai: bool = True,
) -> ResolvedQuery:
    """Resolve a free-text disaster query into a structured ResolvedQuery."""
    raw = query or ""
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return ResolvedQuery(
            intent="unsupported",
            location=None,
            raw=raw,
            cleaned=cleaned,
            resolution_method="none",
        )

    lower_words = set(re.findall(r"[a-z]+", cleaned.lower()))
    if lower_words & {"hello", "hi", "hey"} and len(cleaned.split()) <= 5:
        return ResolvedQuery(
            intent="smalltalk",
            location=None,
            raw=raw,
            cleaned=cleaned,
            resolution_method="none",
        )
    if any(w in cleaned.lower() for w in ["remind", "followup", "again", "what was"]):
        return ResolvedQuery(
            intent="followup",
            location=None,
            raw=raw,
            cleaned=cleaned,
            resolution_method="none",
        )

    if _is_conceptual_or_injected(cleaned):
        return ResolvedQuery(
            intent="investigate",
            location=None,
            time_hint=_extract_time_hint(cleaned),
            hazard_hint=_extract_hazard_hint(cleaned),
            confidence=0.0,
            resolution_method="none",
            raw=raw,
            cleaned=cleaned,
        )

    is_offline = offline if offline is not None else (os.getenv("DISASTERRES_OFFLINE") == "1")

    ai_result: dict[str, Any] | None = None
    if use_ai and not is_offline:
        ai_result = extract_query_entities(cleaned, offline=is_offline)

    if ai_result is not None:
        enriched = _enrich_with_gazetteer(ai_result)
        det_fallback = _deterministic_extract_toponym(cleaned)
        canonical = enriched.get("canonical_query") or enriched.get("locality") or enriched.get("country")
        return ResolvedQuery(
            intent="investigate",
            location=canonical,
            time_hint=enriched.get("time_hint") or det_fallback.get("time_hint"),
            hazard_hint=enriched.get("hazard_hint") or det_fallback.get("hazard_hint"),
            locality=enriched.get("locality"),
            admin_area=enriched.get("admin_area"),
            country=enriched.get("country"),
            country_iso2=enriched.get("country_iso2"),
            canonical_query=canonical,
            confidence=float(enriched.get("confidence", 0.85)),
            resolution_method="ai" if canonical else "none",
            raw=raw,
            cleaned=cleaned,
        )

    det = _deterministic_extract_toponym(cleaned)
    return ResolvedQuery(
        intent="investigate",
        location=det["canonical_query"],
        time_hint=det["time_hint"],
        hazard_hint=det["hazard_hint"],
        locality=det["locality"],
        admin_area=det["admin_area"],
        country=det["country"],
        country_iso2=det["country_iso2"],
        canonical_query=det["canonical_query"],
        confidence=float(det["confidence"]),
        resolution_method=det["resolution_method"],
        raw=raw,
        cleaned=cleaned,
    )
