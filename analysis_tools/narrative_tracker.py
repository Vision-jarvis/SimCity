"""
Cross-platform narrative-transfer detection.

Clusters events from multiple platforms into *narratives* by content
similarity, then reconstructs how each narrative propagates across platforms
over time (e.g. Reddit -> Hacker News -> News/GDELT), with the time lag at each
hop and a measure of how much the content *mutates* as it jumps platforms.

This realizes the "Narrative Tracking" capability and the Phase 2 milestone
"cross-platform narrative transfer detection (Reddit -> HN -> News)".

Similarity is pluggable: by default it uses a dependency-free token-Jaccard
measure (deterministic, runs in CI). Pass ``sim_fn`` (e.g. a
sentence-transformers cosine) for semantic clustering in production.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    0: "reddit", 1: "hackernews", 2: "gdelt", 3: "rss",
    4: "youtube", 5: "wikipedia", 6: "bluesky",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "as", "at", "by", "it", "this", "that",
    "with", "from", "new", "says", "amp",
}


def _tokenize(text: str) -> set:
    return {t for t in _TOKEN_RE.findall((text or "").lower())
            if len(t) > 2 and t not in _STOPWORDS}


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity in [0, 1]."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _platform_name(p) -> str:
    if isinstance(p, str):
        return p
    return PLATFORM_NAMES.get(int(p), "unknown") if p is not None else "unknown"


@dataclass
class NarrativeCluster:
    """A narrative and its cross-platform propagation."""

    id: str
    summary: str
    platforms: List[str] = field(default_factory=list)
    transfer_path: List[Dict] = field(default_factory=list)  # ordered hops
    first_seen: float = 0.0
    first_platform: str = ""
    event_count: int = 0
    mutation_score: float = 0.0
    virality_score: float = 0.0
    related_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class NarrativeTracker:
    """Greedy online clustering + cross-platform transfer reconstruction."""

    def __init__(
        self,
        similarity_threshold: float = 0.25,
        sim_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.sim_fn = sim_fn or jaccard_similarity

    def detect(self, events: List[Dict]) -> List[NarrativeCluster]:
        """Detect narratives across a batch of events.

        Each event needs ``content``, ``platform`` (int or str), ``timestamp``.
        Returns clusters sorted by virality (most viral first).
        """
        # Process chronologically so transfer paths are causally ordered.
        ordered = sorted(events, key=lambda e: e.get("timestamp", 0.0))
        clusters: List[Dict] = []

        for ev in ordered:
            content = ev.get("content", "")
            if not content.strip():
                continue
            best_idx, best_sim = -1, 0.0
            for i, c in enumerate(clusters):
                sim = self.sim_fn(content, c["representative"])
                if sim > best_sim:
                    best_idx, best_sim = i, sim
            if best_idx >= 0 and best_sim >= self.similarity_threshold:
                clusters[best_idx]["events"].append(ev)
            else:
                clusters.append({"representative": content, "events": [ev]})

        result = [self._build_cluster(f"narr_{i:03d}", c) for i, c in enumerate(clusters)]
        result.sort(key=lambda c: c.virality_score, reverse=True)
        return result

    def detect_transfers(self, events: List[Dict]) -> List[NarrativeCluster]:
        """Only narratives that actually crossed >= 2 platforms."""
        return [c for c in self.detect(events) if len(c.platforms) > 1]

    # ------------------------------------------------------------------ #
    def _build_cluster(self, cid: str, raw: Dict) -> NarrativeCluster:
        evs = raw["events"]
        # Earliest appearance per platform.
        per_platform_first: Dict[str, float] = {}
        per_platform_text: Dict[str, str] = {}
        for ev in evs:
            pname = _platform_name(ev.get("platform"))
            ts = ev.get("timestamp", 0.0)
            if pname not in per_platform_first or ts < per_platform_first[pname]:
                per_platform_first[pname] = ts
                per_platform_text[pname] = ev.get("content", "")

        # Order platforms by first appearance -> the transfer path.
        ordered_platforms = sorted(per_platform_first.items(), key=lambda kv: kv[1])
        first_seen = ordered_platforms[0][1]
        transfer_path = [
            {
                "platform": pname,
                "first_seen": ts,
                "lag_hours": round((ts - first_seen) / 3600.0, 3),
            }
            for pname, ts in ordered_platforms
        ]

        return NarrativeCluster(
            id=cid,
            summary=raw["representative"][:200],
            platforms=[p for p, _ in ordered_platforms],
            transfer_path=transfer_path,
            first_seen=first_seen,
            first_platform=ordered_platforms[0][0],
            event_count=len(evs),
            mutation_score=self._mutation_score(list(per_platform_text.values())),
            virality_score=self._virality_score(evs, len(ordered_platforms)),
            related_topics=self._top_terms(evs),
        )

    def _mutation_score(self, texts: List[str]) -> float:
        """1 - mean pairwise similarity across per-platform representatives."""
        if len(texts) < 2:
            return 0.0
        sims, n = 0.0, 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims += self.sim_fn(texts[i], texts[j])
                n += 1
        return round(1.0 - (sims / n), 4) if n else 0.0

    def _virality_score(self, evs: List[Dict], n_platforms: int) -> float:
        """Heuristic in [0, 1]: volume, platform spread, and temporal velocity."""
        import math

        volume = min(1.0, math.log1p(len(evs)) / 5.0)
        spread = min(1.0, n_platforms / len(PLATFORM_NAMES))
        ts = sorted(e.get("timestamp", 0.0) for e in evs)
        span_h = max((ts[-1] - ts[0]) / 3600.0, 1e-6) if len(ts) > 1 else 1.0
        velocity = min(1.0, math.log1p(len(evs) / span_h) / 3.0)
        return round(0.4 * volume + 0.35 * spread + 0.25 * velocity, 4)

    def _top_terms(self, evs: List[Dict], k: int = 5) -> List[str]:
        counts: Dict[str, int] = {}
        for ev in evs:
            for tok in _tokenize(ev.get("content", "")):
                counts[tok] = counts.get(tok, 0) + 1
        return [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]]
