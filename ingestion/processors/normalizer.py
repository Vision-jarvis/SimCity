"""
Event normalizer — standardizes raw platform events into a consistent schema.
Handles text cleaning, timestamp normalization, and field validation.
"""

import re
import html
import logging
from typing import Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Platform ID → name mapping
PLATFORM_NAMES = {
    0: "reddit",
    1: "hackernews",
    2: "gdelt",
    3: "rss",
    4: "youtube",
}


class Normalizer:
    """
    Normalizes raw ingested events into a clean, consistent format
    suitable for NLP processing and graph ingestion.
    """

    def __init__(self, max_content_length: int = 5000):
        self.max_content_length = max_content_length
        self.stats = {"processed": 0, "errors": 0}

    def normalize(self, event: Dict) -> Optional[Dict]:
        """
        Normalize a single event. Returns the cleaned event or None on error.
        """
        try:
            self.stats["processed"] += 1

            normalized = {
                "id": str(event.get("id", "")),
                "platform": int(event.get("platform", -1)),
                "platform_name": PLATFORM_NAMES.get(
                    int(event.get("platform", -1)), "unknown"
                ),
                "timestamp": self._normalize_timestamp(event.get("timestamp")),
                "author": self._clean_author(event.get("author", "")),
                "content": self._clean_content(event.get("content", "")),
                "metadata": event.get("metadata", {}),
            }

            # Validate required fields
            if not normalized["id"] or normalized["platform"] < 0:
                logger.warning(f"Skipping event with missing id or platform: {event.get('id')}")
                return None

            return normalized

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Normalization error for event {event.get('id')}: {e}")
            return None

    def _normalize_timestamp(self, ts) -> float:
        """Convert various timestamp formats to Unix epoch float."""
        if ts is None:
            return datetime.now(timezone.utc).timestamp()
        if isinstance(ts, (int, float)):
            # Already epoch — validate reasonable range
            if ts > 1e12:  # Milliseconds
                return ts / 1000.0
            return float(ts)
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp()
            except ValueError:
                pass
        return datetime.now(timezone.utc).timestamp()

    def _clean_content(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate to max length
        if len(text) > self.max_content_length:
            text = text[: self.max_content_length] + "..."

        return text

    def _clean_author(self, author: str) -> str:
        """Clean author/username string."""
        if not author:
            return "[unknown]"
        author = str(author).strip()
        if author.lower() in ["none", "null", "nan", "[deleted]", "deleted"]:
            return "[deleted]"
        return author

    def get_stats(self) -> Dict:
        """Return normalization statistics."""
        return self.stats
