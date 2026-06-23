"""
Bluesky ingester — pulls public posts via the AT Protocol.

Bluesky exposes a free, key-optional public AppView, giving a reliable,
dependable free social signal (X has no free API). Two modes:

- **Public (no credentials):** follows configured accounts via
  ``app.bsky.feed.getAuthorFeed`` (open on the public AppView, no key).
- **Authenticated (optional app password):** additionally runs
  ``app.bsky.feed.searchPosts`` for query-based discovery (search requires
  auth) and gets higher rate limits.

Maps to Platform=6. Free, $0:
- Public AppView: https://public.api.bsky.app (no auth)
- App passwords (optional): https://bsky.app/settings/app-passwords
"""

import time
import logging
from datetime import datetime, timezone

import requests

from ingestion.sources.base import BaseIngester
from ingestion.config import (
    BLUESKY_APPVIEW_URL,
    BLUESKY_PDS_URL,
    BLUESKY_IDENTIFIER,
    BLUESKY_APP_PASSWORD,
    BLUESKY_ACTORS,
    BLUESKY_QUERIES,
)

logger = logging.getLogger(__name__)


class BlueskyIngester(BaseIngester):
    """Polls Bluesky author feeds (public) and search (when authenticated)."""

    def __init__(
        self,
        producer,
        actors=None,
        queries=None,
        poll_interval: int = 90,
        limit: int = 25,
        identifier: str = "",
        app_password: str = "",
    ):
        super().__init__(producer)
        self.actors = actors or BLUESKY_ACTORS
        self.queries = queries or BLUESKY_QUERIES
        self.poll_interval = poll_interval
        self.limit = limit
        self.identifier = identifier or BLUESKY_IDENTIFIER
        self.app_password = app_password or BLUESKY_APP_PASSWORD
        self.session = requests.Session()
        self._access_jwt = None
        self.seen_ids = set()

    # ------------------------------------------------------------------ #
    def _authenticate(self):
        """Optional: exchange an app password for an access JWT (enables search)."""
        if not (self.identifier and self.app_password):
            return False
        try:
            resp = self.session.post(
                f"{BLUESKY_PDS_URL}/xrpc/com.atproto.server.createSession",
                json={"identifier": self.identifier, "password": self.app_password},
                timeout=10,
            )
            resp.raise_for_status()
            self._access_jwt = resp.json().get("accessJwt")
            logger.info("Bluesky authenticated as %s", self.identifier)
            return True
        except Exception as e:
            logger.warning("Bluesky auth failed (%s); using public reads only", e)
            self._access_jwt = None
            return False

    def _xrpc_get(self, method: str, params: dict, require_auth: bool = False):
        """GET an XRPC method, returning parsed JSON or None on error.

        Authenticated calls go to the PDS; public reads to the AppView.
        """
        headers = {}
        if self._access_jwt:
            base = BLUESKY_PDS_URL
            headers["Authorization"] = f"Bearer {self._access_jwt}"
        elif require_auth:
            return None  # method needs auth but we have none
        else:
            base = BLUESKY_APPVIEW_URL
        try:
            resp = self.session.get(
                f"{base}/xrpc/{method}", params=params, headers=headers, timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Bluesky %s failed (%s): %s", method, params, e)
            return None

    @staticmethod
    def _parse_timestamp(value: str) -> float:
        """Parse an AT Proto ISO-8601 timestamp to a unix epoch (seconds)."""
        if not value:
            return time.time()
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            return time.time()

    def _parse_post(self, post: dict, source: str):
        """Convert one AT Proto post view into a unified event, or None if seen."""
        uri = post.get("uri", "")
        if not uri or uri in self.seen_ids:
            return None
        self.seen_ids.add(uri)
        author = post.get("author", {})
        record = post.get("record", {})
        return self.format_event(
            event_id=f"bsky_{uri.rsplit('/', 1)[-1]}",
            platform=6,  # 6 = Bluesky
            timestamp=self._parse_timestamp(record.get("createdAt", "")),
            author=author.get("handle", "unknown"),
            content=(record.get("text", "") or "").strip(),
            metadata={
                "source": source,
                "uri": uri,
                "like_count": post.get("likeCount", 0),
                "repost_count": post.get("repostCount", 0),
                "reply_count": post.get("replyCount", 0),
            },
        )

    def _parse_author_feed(self, data, actor: str):
        """Parse an `app.bsky.feed.getAuthorFeed` payload (public)."""
        events = []
        for item in (data or {}).get("feed", []):
            ev = self._parse_post(item.get("post", {}), f"@{actor}")
            if ev:
                events.append(ev)
        return events

    def _parse_search(self, data, query: str):
        """Parse an `app.bsky.feed.searchPosts` payload (authenticated)."""
        events = []
        for post in (data or {}).get("posts", []):
            ev = self._parse_post(post, query)
            if ev:
                events.append(ev)
        return events

    def fetch_latest(self):
        """Poll author feeds (always) and search (when authenticated)."""
        self._authenticate()  # no-op without credentials
        mode = "authenticated (feeds + search)" if self._access_jwt else "public (feeds only)"
        logger.info("Bluesky Ingester starting — %s", mode)
        while True:
            # Public: author feeds for configured accounts.
            for actor in self.actors:
                data = self._xrpc_get(
                    "app.bsky.feed.getAuthorFeed", {"actor": actor, "limit": self.limit}
                )
                for event in self._parse_author_feed(data, actor):
                    yield event

            # Authenticated only: query-based search.
            if self._access_jwt:
                for query in self.queries:
                    data = self._xrpc_get(
                        "app.bsky.feed.searchPosts",
                        {"q": query, "limit": self.limit},
                        require_auth=True,
                    )
                    for event in self._parse_search(data, query):
                        yield event

            if len(self.seen_ids) > 100000:
                self.seen_ids = set(list(self.seen_ids)[-50000:])

            time.sleep(self.poll_interval)
