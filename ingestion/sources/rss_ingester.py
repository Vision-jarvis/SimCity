"""
RSS feed ingester for monitoring public news and blog feeds.
Uses feedparser to poll configurable RSS/Atom feeds.
"""

import time
import hashlib
import logging
import feedparser
from typing import List, Optional
from ingestion.sources.base import BaseIngester

logger = logging.getLogger(__name__)


class RSSIngester(BaseIngester):
    """
    Polls RSS/Atom feeds for new articles and converts them to
    the unified event format. Maps to Platform=3.
    """

    def __init__(self, producer, feed_urls: Optional[List[str]] = None, poll_interval: int = 300):
        super().__init__(producer)
        self.feed_urls = feed_urls or [
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://www.reddit.com/r/technology/.rss",
            "https://hnrss.org/frontpage",
        ]
        self.poll_interval = poll_interval
        self.seen_ids = set()

    def _generate_id(self, entry) -> str:
        """Generate a deterministic ID for a feed entry."""
        unique_str = entry.get("id", "") or entry.get("link", "") or entry.get("title", "")
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _parse_timestamp(self, entry) -> float:
        """Extract timestamp from feed entry."""
        for field in ["published_parsed", "updated_parsed"]:
            parsed = entry.get(field)
            if parsed:
                try:
                    return time.mktime(parsed)
                except (TypeError, OverflowError):
                    pass
        return time.time()

    def fetch_latest(self):
        """
        Continuously polls all configured RSS feeds for new entries.
        Yields unified event dicts.
        """
        logger.info(f"RSS Ingester starting with {len(self.feed_urls)} feeds")

        while True:
            for feed_url in self.feed_urls:
                try:
                    feed = feedparser.parse(feed_url)
                    feed_title = feed.feed.get("title", feed_url)

                    for entry in feed.entries:
                        entry_id = self._generate_id(entry)
                        if entry_id in self.seen_ids:
                            continue

                        self.seen_ids.add(entry_id)

                        # Extract content
                        content = entry.get("summary", "")
                        title = entry.get("title", "")
                        full_content = f"{title}\n{content}".strip()

                        metadata = {
                            "feed": feed_title,
                            "feed_url": feed_url,
                            "link": entry.get("link", ""),
                            "tags": [t.get("term", "") for t in entry.get("tags", [])],
                        }

                        yield self.format_event(
                            event_id=f"rss_{entry_id}",
                            platform=3,  # 3 = RSS
                            timestamp=self._parse_timestamp(entry),
                            author=entry.get("author", feed_title),
                            content=full_content,
                            metadata=metadata,
                        )

                except Exception as e:
                    logger.error(f"Error parsing RSS feed {feed_url}: {e}")

            # Trim seen_ids to prevent unbounded memory growth
            if len(self.seen_ids) > 100000:
                self.seen_ids = set(list(self.seen_ids)[-50000:])

            time.sleep(self.poll_interval)
