"""SQLite cache for event deduplication and Twitter search caps."""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from config.agent_config import CACHE_PATH, DAILY_SEARCH_CAP


class AgentCache:
    def __init__(self, db_path: str = CACHE_PATH):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS processed_events (
                event_id TEXT PRIMARY KEY,
                disaster_class TEXT,
                processed_at TEXT,
                images_collected INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS search_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                searched_at TEXT
            );
            """
        )
        self.conn.commit()

    def already_processed(self, event_id: str, allow_zero_retry: bool = False) -> bool:
        row = self.conn.execute(
            "SELECT processed_at, images_collected FROM processed_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return False

        processed_at = _parse_iso(row["processed_at"])
        if processed_at is None:
            return False

        recent = datetime.utcnow() - processed_at < timedelta(hours=24)
        if recent and allow_zero_retry and int(row["images_collected"] or 0) == 0:
            return False
        return recent

    def mark_processed(self, event_id: str, disaster_class: str, images_collected: int) -> None:
        self.conn.execute(
            """
            INSERT INTO processed_events (event_id, disaster_class, processed_at, images_collected)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                disaster_class = excluded.disaster_class,
                processed_at = excluded.processed_at,
                images_collected = excluded.images_collected
            """,
            (event_id, disaster_class, datetime.utcnow().isoformat(), int(images_collected)),
        )
        self.conn.commit()

    def searches_today(self) -> int:
        today = datetime.utcnow().date().isoformat()
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM search_log WHERE substr(searched_at, 1, 10) = ?",
            (today,),
        ).fetchone()
        return int(row["count"] if row else 0)

    def can_search_today(self) -> bool:
        return self.searches_today() < DAILY_SEARCH_CAP

    def query_was_searched(self, query: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM search_log WHERE query = ? LIMIT 1",
            (query,),
        ).fetchone()
        return row is not None

    def log_search(self, query: str) -> None:
        self.conn.execute(
            "INSERT INTO search_log (query, searched_at) VALUES (?, ?)",
            (query, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None
