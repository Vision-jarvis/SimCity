"""Strawberry GraphQL schema for the SimCity API."""

import strawberry
from typing import List, Optional
import time


@strawberry.type
class GraphNode:
    id: str
    label: str
    platform: Optional[str] = None
    influence_score: Optional[float] = None


@strawberry.type
class GraphEdge:
    source: str
    target: str
    type: str
    weight: Optional[float] = None


@strawberry.type
class TrendInfo:
    topic: str
    score: float
    platform: str
    sentiment: str


@strawberry.type
class NarrativeInfo:
    id: str
    summary: str
    platforms: List[str]
    virality_score: float
    sentiment_avg: float


@strawberry.type
class SimulationStep:
    t: float
    S: float
    E: float
    I: float
    R: float
    Z: float
    D: float


@strawberry.type
class Query:
    @strawberry.field
    def nodes(self, label: Optional[str] = None, limit: int = 50) -> List[GraphNode]:
        """Query graph nodes."""
        # Mock data — in production queries Neo4j
        return [
            GraphNode(id=f"node_{i}", label=label or "Author", platform="reddit")
            for i in range(min(limit, 10))
        ]

    @strawberry.field
    def edges(self, type: Optional[str] = None, limit: int = 50) -> List[GraphEdge]:
        """Query graph edges."""
        return [
            GraphEdge(source=f"node_{i}", target=f"node_{i+1}", type=type or "POSTED")
            for i in range(min(limit, 5))
        ]

    @strawberry.field
    def trends(self, limit: int = 10) -> List[TrendInfo]:
        """Get current trending topics."""
        return [
            TrendInfo(topic="AI Safety", score=0.95, platform="reddit", sentiment="negative"),
            TrendInfo(topic="Climate Summit", score=0.87, platform="gdelt", sentiment="neutral"),
        ][:limit]

    @strawberry.field
    def narratives(self, limit: int = 5) -> List[NarrativeInfo]:
        """Get active narratives."""
        return [
            NarrativeInfo(
                id="narr_001",
                summary="AI regulation debate across platforms",
                platforms=["reddit", "hackernews"],
                virality_score=0.88,
                sentiment_avg=-0.15,
            ),
        ][:limit]


schema = strawberry.Schema(query=Query)
