"""
Wikipedia ingester — polls the Wikimedia recent-changes API for live edit
activity. Completely free, no API key required.

Recent edits are a strong real-time signal of what the internet is collectively
paying attention to (breaking events spike edit volume on the relevant pages),
which feeds the knowledge graph and narrative tracker. Maps to Platform=5.
"""

import time
import logging
import requests

from ingestion.sources.base import BaseIngester
from ingestion.config import WIKIPEDIA_API_URL

logger = logging.getLogger(__name__)


class WikipediaIngester(BaseIngester):
    """Polls `list=recentchanges` for new edits and yields unified events."""

    def __init__(self, producer, poll_interval: int = 30, limit: int = 50):
        super().__init__(producer)
        self.poll_interval = poll_interval
        self.limit = limit
        self.session = requests.Session()
        self.last_rcid = None

    def _fetch_changes(self):
        """Fetch the latest recent changes as parsed JSON, or None on error."""
        params = {
            "action": "query",
            "list": "recentchanges",
            "rcprop": "title|timestamp|comment|user|ids|sizes",
            "rclimit": self.limit,
            "rctype": "edit|new",
            "format": "json",
        }
        try:
            resp = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch Wikipedia recent changes: {e}")
            return None

    def _parse_changes(self, data):
        """Convert a recent-changes API payload into unified event dicts.

        Separated from network I/O so it is unit-testable with fixture JSON.
        """
        events = []
        changes = (data or {}).get("query", {}).get("recentchanges", [])
        for rc in changes:
            rcid = rc.get("rcid")
            # Skip anything we've already emitted in a prior poll.
            if self.last_rcid is not None and rcid is not None and rcid <= self.last_rcid:
                continue

            ts = rc.get("timestamp", "")
            try:
                # Wikimedia timestamps are ISO-8601 Zulu, e.g. 2024-01-02T03:04:05Z
                epoch = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
            except (ValueError, TypeError):
                epoch = time.time()

            title = rc.get("title", "")
            comment = rc.get("comment", "")
            delta = rc.get("newlen", 0) - rc.get("oldlen", 0)

            events.append(
                self.format_event(
                    event_id=f"wiki_{rcid}",
                    platform=5,  # 5 = Wikipedia
                    timestamp=epoch,
                    author=rc.get("user", "anonymous"),
                    content=f"{title}: {comment}".strip(": ").strip(),
                    metadata={
                        "title": title,
                        "type": rc.get("type", "edit"),
                        "size_delta": delta,
                        "pageid": rc.get("pageid"),
                        "rcid": rcid,
                    },
                )
            )

        # Advance the cursor to the highest rcid we saw.
        rcids = [c.get("rcid") for c in changes if c.get("rcid") is not None]
        if rcids:
            self.last_rcid = max(rcids)
        return events

    def fetch_latest(self):
        """Continuously poll Wikimedia recent changes and yield events."""
        logger.info("Wikipedia Ingester starting (recent changes)")
        while True:
            data = self._fetch_changes()
            for event in self._parse_changes(data):
                yield event
            time.sleep(self.poll_interval)
