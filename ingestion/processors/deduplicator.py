"""
Event deduplicator using content hashing and bloom-filter-like seen tracking.
Prevents duplicate events from reaching the graph and NLP pipelines.
"""

import hashlib
import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Deduplicates events using a combination of:
    1. Exact ID matching (event_id)
    2. Content-based fingerprinting (hash of normalized content)
    """

    def __init__(self, max_seen: int = 500000):
        self.max_seen = max_seen
        self.seen_ids: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.stats = {"total": 0, "duplicates": 0, "passed": 0}

    def _content_hash(self, text: str) -> str:
        """Generate a fingerprint from normalized content."""
        normalized = text.lower().strip()
        # Remove common noise
        for char in ["\n", "\r", "\t", "  "]:
            normalized = normalized.replace(char, " ")
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def is_duplicate(self, event: Dict) -> bool:
        """
        Check if an event is a duplicate.

        Args:
            event: Unified event dict with 'id' and 'content' fields.

        Returns:
            True if duplicate, False if new.
        """
        self.stats["total"] += 1

        event_id = event.get("id", "")
        content = event.get("content", "")

        # Check 1: Exact ID match
        if event_id and event_id in self.seen_ids:
            self.stats["duplicates"] += 1
            return True

        # Check 2: Content fingerprint
        if content:
            content_hash = self._content_hash(content)
            if content_hash in self.seen_hashes:
                self.stats["duplicates"] += 1
                return True
            self.seen_hashes.add(content_hash)

        # Mark as seen
        if event_id:
            self.seen_ids.add(event_id)

        # Prevent unbounded memory growth
        self._trim()

        self.stats["passed"] += 1
        return False

    def process(self, event: Dict) -> Optional[Dict]:
        """
        Process an event — returns the event if new, None if duplicate.
        """
        if self.is_duplicate(event):
            return None
        return event

    def _trim(self):
        """Trim seen sets when they exceed max size."""
        if len(self.seen_ids) > self.max_seen:
            trim_to = self.max_seen // 2
            self.seen_ids = set(list(self.seen_ids)[-trim_to:])
            logger.info(f"Trimmed seen_ids to {len(self.seen_ids)}")

        if len(self.seen_hashes) > self.max_seen:
            trim_to = self.max_seen // 2
            self.seen_hashes = set(list(self.seen_hashes)[-trim_to:])
            logger.info(f"Trimmed seen_hashes to {len(self.seen_hashes)}")

    def get_stats(self) -> Dict:
        """Return deduplication statistics."""
        return {
            **self.stats,
            "dedup_rate": (
                round(self.stats["duplicates"] / max(self.stats["total"], 1), 4)
            ),
        }
