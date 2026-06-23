"""Tests for ingestion sources and processors."""



class TestBaseIngester:
    def test_format_event(self):
        from ingestion.sources.base import BaseIngester
        # format_event is a @staticmethod; call it directly since
        # BaseIngester is abstract and cannot be instantiated.
        event = BaseIngester.format_event(
            event_id="test_001",
            platform=0,
            timestamp=1700000000.0,
            author="test_user",
            content="Hello world",
            metadata={"score": 42},
        )
        assert event["id"] == "test_001"
        assert event["platform"] == 0
        assert event["author"] == "test_user"
        assert event["content"] == "Hello world"
        assert event["metadata"]["score"] == 42


class TestDeduplicator:
    def test_basic_dedup(self):
        from ingestion.processors.deduplicator import Deduplicator
        dedup = Deduplicator()

        event = {"id": "evt_1", "content": "Hello world"}
        assert not dedup.is_duplicate(event)
        assert dedup.is_duplicate(event)  # Second time is duplicate

    def test_content_hash_dedup(self):
        from ingestion.processors.deduplicator import Deduplicator
        dedup = Deduplicator()

        e1 = {"id": "a", "content": "Same content here"}
        e2 = {"id": "b", "content": "Same content here"}  # Different ID, same content

        assert not dedup.is_duplicate(e1)
        assert dedup.is_duplicate(e2)

    def test_process_returns_none_for_dup(self):
        from ingestion.processors.deduplicator import Deduplicator
        dedup = Deduplicator()

        event = {"id": "x", "content": "test"}
        assert dedup.process(event) is not None
        assert dedup.process(event) is None

    def test_stats(self):
        from ingestion.processors.deduplicator import Deduplicator
        dedup = Deduplicator()

        dedup.process({"id": "1", "content": "a"})
        dedup.process({"id": "1", "content": "a"})
        dedup.process({"id": "2", "content": "b"})

        stats = dedup.get_stats()
        assert stats["total"] == 3
        assert stats["duplicates"] == 1
        assert stats["passed"] == 2


class TestNormalizer:
    def test_basic_normalize(self):
        from ingestion.processors.normalizer import Normalizer
        norm = Normalizer()

        event = {
            "id": "test_1",
            "platform": 0,
            "timestamp": 1700000000.0,
            "author": "user",
            "content": "<b>Hello</b> &amp; world",
        }
        result = norm.normalize(event)
        assert result is not None
        assert result["platform_name"] == "reddit"
        assert "&amp;" not in result["content"]
        assert "<b>" not in result["content"]

    def test_missing_id_returns_none(self):
        from ingestion.processors.normalizer import Normalizer
        norm = Normalizer()
        result = norm.normalize({"platform": 0})
        assert result is None

    def test_deleted_author(self):
        from ingestion.processors.normalizer import Normalizer
        norm = Normalizer()
        event = {"id": "x", "platform": 0, "author": "[deleted]"}
        result = norm.normalize(event)
        assert result["author"] == "[deleted]"

    def test_timestamp_milliseconds(self):
        from ingestion.processors.normalizer import Normalizer
        norm = Normalizer()
        event = {"id": "x", "platform": 0, "timestamp": 1700000000000}
        result = norm.normalize(event)
        assert result["timestamp"] == 1700000000.0


class TestWikipediaIngester:
    def test_parse_changes(self):
        from ingestion.sources.wikipedia_ingester import WikipediaIngester
        ing = WikipediaIngester(producer=None)
        payload = {
            "query": {
                "recentchanges": [
                    {
                        "rcid": 101, "type": "edit", "title": "Climate change",
                        "user": "EditorBot", "comment": "fix typo",
                        "timestamp": "2024-01-02T03:04:05Z",
                        "oldlen": 100, "newlen": 150, "pageid": 7,
                    },
                    {
                        "rcid": 102, "type": "new", "title": "Breaking Event",
                        "user": "Reporter", "comment": "created",
                        "timestamp": "2024-01-02T03:05:05Z",
                        "oldlen": 0, "newlen": 500, "pageid": 8,
                    },
                ]
            }
        }
        events = ing._parse_changes(payload)
        assert len(events) == 2
        assert events[0]["platform"] == 5
        assert events[0]["id"] == "wiki_101"
        assert events[0]["metadata"]["size_delta"] == 50
        assert "Climate change" in events[0]["content"]
        # Cursor advanced -> re-parsing the same payload yields nothing new.
        assert ing._parse_changes(payload) == []

    def test_parse_empty(self):
        from ingestion.sources.wikipedia_ingester import WikipediaIngester
        ing = WikipediaIngester(producer=None)
        assert ing._parse_changes(None) == []
        assert ing._parse_changes({}) == []


class TestBlueskyIngester:
    def test_parse_author_feed(self):
        """Public mode parses getAuthorFeed payloads (no key needed)."""
        from ingestion.sources.bluesky_ingester import BlueskyIngester
        ing = BlueskyIngester(producer=None)
        payload = {
            "feed": [
                {"post": {
                    "uri": "at://did:plc:abc/app.bsky.feed.post/xyz1",
                    "author": {"handle": "nytimes.com"},
                    "record": {"text": "breaking news about AI", "createdAt": "2024-01-02T03:04:05.000Z"},
                    "likeCount": 5, "repostCount": 2, "replyCount": 1,
                }},
                {"post": {
                    "uri": "at://did:plc:def/app.bsky.feed.post/xyz2",
                    "author": {"handle": "bbc.com"},
                    "record": {"text": "more breaking news", "createdAt": "2024-01-02T03:05:05.000Z"},
                }},
            ]
        }
        events = ing._parse_author_feed(payload, "nytimes.com")
        assert len(events) == 2
        assert events[0]["platform"] == 6
        assert events[0]["id"] == "bsky_xyz1"
        assert events[0]["author"] == "nytimes.com"
        assert events[0]["metadata"]["like_count"] == 5
        assert events[0]["metadata"]["source"] == "@nytimes.com"
        # Same payload again -> deduped via uri.
        assert ing._parse_author_feed(payload, "nytimes.com") == []

    def test_parse_search_authenticated_shape(self):
        from ingestion.sources.bluesky_ingester import BlueskyIngester
        ing = BlueskyIngester(producer=None)
        payload = {"posts": [{
            "uri": "at://did:plc:abc/app.bsky.feed.post/s1",
            "author": {"handle": "alice.bsky.social"},
            "record": {"text": "misinfo claim", "createdAt": "2024-01-02T03:04:05.000Z"},
        }]}
        events = ing._parse_search(payload, "misinformation")
        assert len(events) == 1
        assert events[0]["metadata"]["source"] == "misinformation"

    def test_parse_timestamp_and_empty(self):
        from ingestion.sources.bluesky_ingester import BlueskyIngester
        ing = BlueskyIngester(producer=None)
        assert ing._parse_timestamp("2024-01-02T03:04:05.000Z") > 0
        assert ing._parse_author_feed(None, "x") == []
        assert ing._parse_search(None, "q") == []


class TestEnricher:
    def test_enricher_no_models(self):
        """Enricher should work gracefully when NLP models aren't available."""
        from ingestion.processors.enricher import Enricher
        enricher = Enricher(
            enable_embeddings=False,
            enable_sentiment=False,
            enable_toxicity=False,
            enable_misinfo=False,
        )
        event = {"id": "x", "content": "test content"}
        result = enricher.enrich(event)
        assert "nlp" in result
        assert isinstance(result["nlp"], dict)
