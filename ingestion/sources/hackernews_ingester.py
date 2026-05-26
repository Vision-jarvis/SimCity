import time
import requests
import logging
from ingestion.sources.base import BaseIngester
from ingestion.config import HN_API_BASE_URL

logger = logging.getLogger(__name__)

class HackerNewsIngester(BaseIngester):
    """
    Polls the official Hacker News Firebase API for the latest items.
    Maps Hacker News data to Platform=1.
    """
    
    def __init__(self, producer, poll_interval=10):
        super().__init__(producer)
        self.poll_interval = poll_interval
        self.last_item_id = None
        self.session = requests.Session()

    def _get_max_item(self):
        try:
            resp = self.session.get(f"{HN_API_BASE_URL}/maxitem.json", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch HN maxitem: {e}")
            return None

    def _fetch_item(self, item_id):
        try:
            resp = self.session.get(f"{HN_API_BASE_URL}/item/{item_id}.json", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch HN item {item_id}: {e}")
            return None

    def fetch_latest(self):
        """
        Continuously polls for new items up to the maxitem.
        Yields them as unified event dictionaries.
        """
        # Initialize last_item_id
        if self.last_item_id is None:
            max_id = self._get_max_item()
            if max_id:
                self.last_item_id = max_id
                logger.info(f"HN Ingester initialized at max item {self.last_item_id}")
            else:
                self.last_item_id = 0

        while True:
            current_max = self._get_max_item()
            if not current_max or current_max <= self.last_item_id:
                time.sleep(self.poll_interval)
                continue

            # Fetch all items from last_item_id+1 to current_max
            for item_id in range(self.last_item_id + 1, current_max + 1):
                item = self._fetch_item(item_id)
                if not item:
                    continue
                
                # We only care about stories and comments for the knowledge graph
                item_type = item.get("type", "")
                if item_type not in ["story", "comment"]:
                    continue

                content = item.get("text", "")
                if item_type == "story":
                    content = f"{item.get('title', '')}\n{content}"

                metadata = {
                    "type": item_type,
                    "score": item.get("score", 0),
                    "parent": item.get("parent"),
                    "url": item.get("url")
                }

                yield self.format_event(
                    event_id=f"hn_{item['id']}",
                    platform=1, # 1 = Hacker News
                    timestamp=item.get("time", time.time()),
                    author=item.get("by", "[deleted]"),
                    content=content.strip(),
                    metadata=metadata
                )
            
            self.last_item_id = current_max
            time.sleep(self.poll_interval)
