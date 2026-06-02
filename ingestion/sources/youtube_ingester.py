"""
YouTube Data API v3 ingester for monitoring video trends and comments.
Uses 10,000 free quota units/day.
"""

import time
import logging
import os
from typing import List, Optional
from ingestion.sources.base import BaseIngester

logger = logging.getLogger(__name__)


class YouTubeIngester(BaseIngester):
    """
    Polls YouTube Data API v3 for trending/search videos and their comments.
    Maps to Platform=4. Quota-aware: each search costs 100 units,
    video details cost 1 unit, comments cost 1 unit.
    """

    def __init__(
        self,
        producer,
        api_key: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        poll_interval: int = 3600,  # 1 hour default (quota conservation)
        max_results: int = 10,
    ):
        super().__init__(producer)
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
        self.search_queries = search_queries or [
            "breaking news",
            "trending technology",
            "viral internet",
        ]
        self.poll_interval = poll_interval
        self.max_results = max_results
        self.seen_ids = set()
        self._service = None

    @property
    def service(self):
        """Lazy-load YouTube API client."""
        if self._service is None:
            try:
                from googleapiclient.discovery import build

                if not self.api_key:
                    raise ValueError("YOUTUBE_API_KEY not set. Cannot initialize YouTube ingester.")

                self._service = build("youtube", "v3", developerKey=self.api_key)
                logger.info("YouTube API client initialized.")
            except ImportError:
                raise ImportError(
                    "google-api-python-client is required. "
                    "Install with: pip install google-api-python-client"
                )
        return self._service

    def _search_videos(self, query: str) -> list:
        """Search for recent videos matching a query. Costs 100 quota units."""
        try:
            request = self.service.search().list(
                q=query,
                part="snippet",
                type="video",
                order="date",
                maxResults=self.max_results,
                publishedAfter=None,  # Could filter by time
            )
            response = request.execute()
            return response.get("items", [])
        except Exception as e:
            logger.error(f"YouTube search failed for '{query}': {e}")
            return []

    def _get_video_stats(self, video_id: str) -> dict:
        """Get view count, like count, comment count. Costs 1 quota unit."""
        try:
            request = self.service.videos().list(
                id=video_id, part="statistics"
            )
            response = request.execute()
            items = response.get("items", [])
            if items:
                return items[0].get("statistics", {})
        except Exception as e:
            logger.error(f"Failed to get stats for video {video_id}: {e}")
        return {}

    def fetch_latest(self):
        """
        Continuously polls YouTube for new videos matching search queries.
        Yields unified event dicts.
        """
        if not self.api_key:
            logger.warning("YouTube API key not set. Skipping YouTube ingestion.")
            return

        logger.info(f"YouTube Ingester starting with queries: {self.search_queries}")

        while True:
            for query in self.search_queries:
                results = self._search_videos(query)

                for item in results:
                    video_id = item.get("id", {}).get("videoId", "")
                    if not video_id or video_id in self.seen_ids:
                        continue

                    self.seen_ids.add(video_id)
                    snippet = item.get("snippet", {})

                    # Get engagement stats
                    stats = self._get_video_stats(video_id)

                    metadata = {
                        "video_id": video_id,
                        "channel": snippet.get("channelTitle", ""),
                        "channel_id": snippet.get("channelId", ""),
                        "view_count": int(stats.get("viewCount", 0)),
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0)),
                        "url": f"https://youtube.com/watch?v={video_id}",
                        "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
                    }

                    # Parse published timestamp
                    try:
                        from datetime import datetime
                        published = snippet.get("publishedAt", "")
                        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                        timestamp = dt.timestamp()
                    except Exception:
                        timestamp = time.time()

                    content = f"{snippet.get('title', '')}\n{snippet.get('description', '')}"

                    yield self.format_event(
                        event_id=f"yt_{video_id}",
                        platform=4,  # 4 = YouTube
                        timestamp=timestamp,
                        author=snippet.get("channelTitle", "[unknown]"),
                        content=content.strip(),
                        metadata=metadata,
                    )

            time.sleep(self.poll_interval)
