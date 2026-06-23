"""Tests for cross-platform narrative-transfer detection."""

from analysis_tools.narrative_tracker import NarrativeTracker, jaccard_similarity


def _ev(content, platform, ts):
    return {"content": content, "platform": platform, "timestamp": ts}


def test_jaccard_basic():
    assert jaccard_similarity("ai safety regulation", "ai safety rules") > 0.2
    assert jaccard_similarity("ai safety", "crypto market crash") == 0.0


def test_clusters_similar_events_across_platforms():
    tracker = NarrativeTracker(similarity_threshold=0.2)
    base = 1_700_000_000.0
    events = [
        _ev("New AI safety regulation proposed by government", 0, base),          # reddit
        _ev("AI safety regulation bill sparks debate online", 1, base + 3600),    # hackernews
        _ev("Government AI safety regulation covered by news outlets", 2, base + 7200),  # gdelt
        _ev("Bitcoin crypto market crash wipes billions", 0, base + 100),         # unrelated
    ]
    narratives = tracker.detect(events)
    # One AI narrative (3 platforms) + one crypto narrative (1 platform).
    ai = max(narratives, key=lambda c: len(c.platforms))
    assert ai.event_count == 3
    assert set(ai.platforms) == {"reddit", "hackernews", "gdelt"}


def test_transfer_path_is_time_ordered():
    tracker = NarrativeTracker(similarity_threshold=0.2)
    base = 1_700_000_000.0
    events = [
        _ev("election disinformation campaign spreads rapidly", 1, base + 7200),   # hn later
        _ev("election disinformation campaign detected on platform", 0, base),     # reddit first
        _ev("election disinformation campaign reported by media", 2, base + 14400),# gdelt last
    ]
    transfers = tracker.detect_transfers(events)
    assert len(transfers) == 1
    path = transfers[0].transfer_path
    assert [hop["platform"] for hop in path] == ["reddit", "hackernews", "gdelt"]
    assert path[0]["lag_hours"] == 0.0
    assert path[1]["lag_hours"] == 2.0
    assert path[2]["lag_hours"] == 4.0
    assert transfers[0].first_platform == "reddit"


def test_single_platform_not_a_transfer():
    tracker = NarrativeTracker(similarity_threshold=0.2)
    events = [
        _ev("local sports team wins championship game", 0, 1.0),
        _ev("local sports team wins championship trophy", 0, 2.0),
    ]
    assert tracker.detect_transfers(events) == []
    # But it is still a (single-platform) narrative with zero mutation.
    narr = tracker.detect(events)
    assert len(narr) == 1
    assert narr[0].mutation_score == 0.0


def test_mutation_score_increases_with_drift():
    tracker = NarrativeTracker(similarity_threshold=0.1)
    base = 1_700_000_000.0
    # Same core topic but increasingly different wording across platforms.
    events = [
        _ev("climate summit agreement reached today", 0, base),
        _ev("climate summit agreement praised reached", 1, base + 3600),
        _ev("climate deal globally protests erupt streets", 2, base + 7200),
    ]
    narr = tracker.detect(events)[0]
    assert 0.0 < narr.mutation_score <= 1.0
