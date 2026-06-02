"""Tests for ingestion sources and processors."""

import pytest
import time


class TestBaseIngester:
    def test_format_event(self):
        from ingestion.sources.base import BaseIngester
        ingester = BaseIngester(producer=None)
        event = ingester.format_event(
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
