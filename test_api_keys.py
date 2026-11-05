"""Test script to verify connectivity and validity of all API credentials configured in .env.

Uses Python's standard library only (urllib, json) so it runs with zero extra dependencies.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and k not in os.environ:
                os.environ[k] = v


env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(env_file)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def _make_request(url: str, data: dict | None = None, headers: dict | None = None, method: str | None = None, timeout: int = 15) -> tuple[int, dict | str]:
    headers = headers or {}
    encoded_data = None
    if data is not None:
        encoded_data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=encoded_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                return response.status, json.loads(body)
            except Exception:
                return response.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body
    except Exception as e:
        raise e


def test_openai() -> None:
    print("\n--- 1. Testing OpenAI API ---")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print(f"{SKIP} OPENAI_API_KEY is not configured.")
        return

    model = os.getenv("LLM_MODEL_SMALL", "gpt-4o-mini").strip()
    try:
        status, res = _make_request(
            "https://api.openai.com/v1/chat/completions",
            data={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        if status == 200:
            print(f"{PASS} OpenAI responded successfully (Model: {model}).")
        else:
            print(f"{FAIL} OpenAI HTTP {status}: {res}")
    except Exception as e:
        print(f"{FAIL} OpenAI request failed: {e}")


def test_google_custom_search() -> None:
    print("\n--- 2. Testing Google Custom Search API ---")
    key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "").strip()
    cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX", "").strip()

    if not key:
        print(f"{SKIP} GOOGLE_CUSTOM_SEARCH_API_KEY is not configured.")
        return
    if not cx:
        print(f"{FAIL} GOOGLE_CUSTOM_SEARCH_API_KEY is set, but GOOGLE_CUSTOM_SEARCH_CX is missing!")
        print("       (Please create a search engine at https://programmablesearchengine.google.com/)")
        return

    try:
        params = urllib.parse.urlencode({
            "key": key,
            "cx": cx,
            "q": "disaster test",
            "searchType": "image",
            "num": 3,
        })
        status, res = _make_request(f"https://customsearch.googleapis.com/customsearch/v1?{params}", timeout=12)
        if status == 200 and isinstance(res, dict):
            items = res.get("items", [])
            print(f"{PASS} Google Custom Search succeeded! Found {len(items)} images.")
        else:
            print(f"{FAIL} Google Custom Search HTTP {status}: {res}")
    except Exception as e:
        print(f"{FAIL} Google Custom Search request failed: {e}")


def test_serper() -> None:
    print("\n--- 3. Testing Serper.dev API ---")
    key = os.getenv("SERPER_API_KEY", "").strip()
    if not key:
        print(f"{SKIP} SERPER_API_KEY is not configured.")
        return

    try:
        status, res = _make_request(
            "https://google.serper.dev/images",
            data={"q": "disaster test", "num": 5},
            headers={"X-API-KEY": key},
            timeout=12,
        )
        if status == 200 and isinstance(res, dict):
            images = res.get("images", [])
            print(f"{PASS} Serper.dev succeeded! Found {len(images)} images.")
        else:
            print(f"{FAIL} Serper.dev HTTP {status}: {res}")
    except Exception as e:
        print(f"{FAIL} Serper.dev request failed: {e}")


def test_serpapi() -> None:
    print("\n--- 4. Testing SerpAPI ---")
    key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not key:
        print(f"{SKIP} SERPAPI_API_KEY is not configured.")
        return

    try:
        params = urllib.parse.urlencode({
            "engine": "google_images",
            "q": "disaster test",
            "api_key": key,
            "num": 5,
        })
        status, res = _make_request(f"https://serpapi.com/search?{params}", timeout=12)
        if status == 200 and isinstance(res, dict):
            images = res.get("images_results", [])
            print(f"{PASS} SerpAPI succeeded! Found {len(images)} images.")
        else:
            print(f"{FAIL} SerpAPI HTTP {status}: {res}")
    except Exception as e:
        print(f"{FAIL} SerpAPI request failed: {e}")


def test_dataforseo() -> None:
    print("\n--- 5. Testing DataForSEO API ---")
    login = os.getenv("DATAFORSEO_LOGIN", "").strip()
    password = os.getenv("DATAFORSEO_PASSWORD", "").strip()
    legacy_key = os.getenv("DATAFORSEO_API_KEY", "").strip()

    auth_header = None
    if login and password:
        auth_header = base64.b64encode(f"{login}:{password}".encode()).decode("utf-8")
    elif legacy_key:
        auth_header = legacy_key

    if not auth_header:
        print(f"{SKIP} DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD is not configured.")
        return

    try:
        payload = [{"keyword": "disaster test", "language_code": "en", "location_code": 2840, "depth": 5}]
        status, res = _make_request(
            "https://api.dataforseo.com/v3/serp/google/images/live/advanced",
            data=payload,
            headers={"Authorization": f"Basic {auth_header}"},
            timeout=15,
        )
        if status == 200 and isinstance(res, dict):
            status_message = res.get("status_message", "Ok.")
            tasks = res.get("tasks", [])
            if tasks and tasks[0].get("status_message") == "Ok.":
                items = tasks[0].get("result", [])[0].get("items", []) if tasks[0].get("result") else []
                print(f"{PASS} DataForSEO succeeded! Found {len(items)} images.")
            else:
                print(f"{FAIL} DataForSEO API response status: {status_message}")
        else:
            print(f"{FAIL} DataForSEO HTTP {status}: {res}")
    except Exception as e:
        print(f"{FAIL} DataForSEO request failed: {e}")


def test_free_fallbacks() -> None:
    print("\n--- 6. Testing Free Fallback Services (Wikimedia Commons & GDELT) ---")
    # Wikimedia
    try:
        params = urllib.parse.urlencode({
            "action": "query",
            "generator": "search",
            "gsrsearch": "earthquake aerial damage",
            "gsrnamespace": "6",
            "gsrlimit": "3",
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        })
        status, res = _make_request(
            f"https://commons.wikimedia.org/w/api.php?{params}",
            headers={"User-Agent": "DisasterRes-Net/1.0 (test@example.com)"},
            timeout=10,
        )
        if status == 200 and isinstance(res, dict):
            pages = res.get("query", {}).get("pages", {})
            print(f"{PASS} Wikimedia Commons: Available and returned {len(pages)} items.")
        else:
            print(f"{FAIL} Wikimedia Commons returned HTTP {status}")
    except Exception as e:
        print(f"{FAIL} Wikimedia Commons test failed: {e}")

    # GDELT
    try:
        params = urllib.parse.urlencode({
            "query": '"earthquake" disaster',
            "mode": "artlist",
            "maxrecords": 5,
            "format": "json",
            "sort": "HybridRel",
        })
        status, res = _make_request(
            f"https://api.gdeltproject.org/api/v2/doc/doc?{params}",
            headers={"User-Agent": "DisasterRes-Net/1.0"},
            timeout=10,
        )
        if status == 200 and isinstance(res, dict):
            articles = res.get("articles", [])
            print(f"{PASS} GDELT API: Available and returned {len(articles)} records.")
        else:
            print(f"{FAIL} GDELT API returned HTTP {status}")
    except Exception as e:
        print(f"{FAIL} GDELT test failed: {e}")


if __name__ == "__main__":
    print("==================================================")
    print("      DisasterRes-Net API Verification Tool       ")
    print("==================================================")
    test_openai()
    test_google_custom_search()
    test_serper()
    test_serpapi()
    test_dataforseo()
    test_free_fallbacks()
    print("\n==================================================")
    print("Verification complete.")
