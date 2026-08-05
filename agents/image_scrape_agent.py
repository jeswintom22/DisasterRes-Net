"""Resolve event locations and collect Twitter image candidates via twikit."""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional

import requests

from agents.cache import AgentCache
from agents.schemas import DisasterEvent, ImageCandidate
from config.agent_config import (
    DAILY_SEARCH_CAP,
    NOMINATIM_DELAY_SECONDS,
    TWITTER_SEARCH_DELAY_SECONDS,
    load_twikit_credentials,
)


NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "DisasterResNet-Research/1.0 (academic project)"}

LOCALIZED_TERMS = {
    "earthquake": ["deprem", "terremoto", "gempa", "lindol"],
    "flood": ["sel", "inundacion", "banjir", "baha"],
    "hurricane": ["kasirga", "huracan", "siklon", "bagyo"],
    "wildfire": ["orman yangini", "incendio forestal", "kebakaran hutan", "sunog"],
    "landslide": ["heyelan", "deslizamiento", "tanah longsor", "pagguho"],
}


class ImageScrapeAgent:
    def __init__(self, cache: AgentCache):
        self.cache = cache
        self._last_nominatim_call = 0.0
        self._last_twitter_search = 0.0

    def collect_for_event(self, event: DisasterEvent) -> List[ImageCandidate]:
        return asyncio.run(self.collect_for_event_async(event))

    async def collect_for_event_async(self, event: DisasterEvent) -> List[ImageCandidate]:
        if not event.place_name:
            event.place_name = self.reverse_geocode(event.latitude, event.longitude)

        credentials = load_twikit_credentials()
        if credentials is None:
            print("[OBSERVATION] Missing TWIKIT_USERNAME/TWIKIT_EMAIL/TWIKIT_PASSWORD; skipping Twitter search.")
            return []

        try:
            from twikit import Client
        except ImportError:
            print("[OBSERVATION] twikit is not installed; skipping Twitter search.")
            return []

        client = Client("en-US")
        cookies_file = os.path.join(".cache", "twikit_cookies.json")
        os.makedirs(os.path.dirname(cookies_file), exist_ok=True)
        await client.login(
            auth_info_1=credentials.username,
            auth_info_2=credentials.email,
            password=credentials.password,
            cookies_file=cookies_file,
        )

        candidates: List[ImageCandidate] = []
        for query in self.build_queries(event):
            if not self.cache.can_search_today():
                print(f"[OBSERVATION] Daily Twitter search cap reached ({DAILY_SEARCH_CAP}); skipping remaining searches.")
                break
            if self.cache.query_was_searched(query):
                print(f"[OBSERVATION] Query already searched in a previous run; skipping: {query}")
                continue
            self._throttle_twitter()
            self.cache.log_search(query)
            tweets = await client.search_tweet(query, "Latest")
            candidates.extend(self._candidates_from_tweets(tweets, event))
        return candidates

    def reverse_geocode(self, lat: float, lon: float) -> str:
        elapsed = time.time() - self._last_nominatim_call
        if elapsed < NOMINATIM_DELAY_SECONDS:
            time.sleep(NOMINATIM_DELAY_SECONDS - elapsed)
        params = {"lat": lat, "lon": lon, "format": "json"}
        response = requests.get(NOMINATIM_URL, params=params, headers=NOMINATIM_HEADERS, timeout=10)
        self._last_nominatim_call = time.time()
        response.raise_for_status()
        address = response.json().get("address", {})
        return (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("state")
            or address.get("country")
            or "unknown"
        )

    def build_queries(self, event: DisasterEvent) -> List[str]:
        place = (event.place_name or event.title or "unknown").split(",")[0].strip()
        place_tag = "".join(ch for ch in place.lower() if ch.isalnum())
        localized = LOCALIZED_TERMS.get(event.disaster_class, [])
        queries = [
            f"{place} {event.disaster_class} damage",
            f"#{place_tag} #{event.disaster_class}",
        ]
        if localized:
            queries.insert(1, f"{place} {localized[0]}")
        return [query for query in queries if query.strip()]

    def _throttle_twitter(self) -> None:
        elapsed = time.time() - self._last_twitter_search
        if elapsed < TWITTER_SEARCH_DELAY_SECONDS:
            time.sleep(TWITTER_SEARCH_DELAY_SECONDS - elapsed)
        self._last_twitter_search = time.time()

    def _candidates_from_tweets(self, tweets: object, event: DisasterEvent) -> List[ImageCandidate]:
        out: List[ImageCandidate] = []
        cutoff = event.event_time - timedelta(hours=1)
        for tweet in tweets or []:
            tweet_time = _parse_tweet_datetime(getattr(tweet, "created_at", None))
            if tweet_time is None or tweet_time < cutoff:
                continue
            tweet_id = str(getattr(tweet, "id", "") or getattr(tweet, "tweet_id", "unknown"))
            for media in getattr(tweet, "media", None) or []:
                url = getattr(media, "media_url", None) or getattr(media, "url", None)
                if not url:
                    continue
                out.append(
                    ImageCandidate(
                        url=str(url),
                        source_tweet_id=tweet_id,
                        tweet_created_at=tweet_time,
                        disaster_class=event.disaster_class,
                        event_id=event.event_id,
                        event_source=event.source,
                        latitude=event.latitude,
                        longitude=event.longitude,
                        place_name=event.place_name,
                    )
                )
        return out


def _parse_tweet_datetime(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if not value:
        return None
    text = str(value)
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=None)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
