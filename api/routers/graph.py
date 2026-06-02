"""Graph query API endpoints (Cypher-backed)."""

from fastapi import APIRouter, HTTPException
from typing import List
from api.schemas.graph_schemas import (
    NodeResponse, EdgeResponse, GraphStatsResponse,
    CommunityResponse, InfluenceResponse, GraphQueryRequest,
)
from api.dependencies import get_graph_client

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/nodes", response_model=List[NodeResponse])
async def get_nodes(label: str = "", limit: int = 100):
    """Get nodes from the knowledge graph, optionally filtered by label."""
    client = get_graph_client()
    if not client:
        # Return mock data when Neo4j is offline
        return [
            NodeResponse(id=f"mock_{i}", label=label or "Author", properties={"name": f"user_{i}"})
            for i in range(min(limit, 10))
        ]

    label_filter = f":{label}" if label else ""
    query = f"MATCH (n{label_filter}) RETURN n LIMIT $limit"
    try:
        results = client.execute_read(query, {"limit": limit})
        return [
            NodeResponse(
                id=str(r["n"].get("id", r["n"].get("name", ""))),
                label=list(r["n"].labels)[0] if hasattr(r["n"], "labels") else "Node",
                properties=dict(r["n"]),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/edges", response_model=List[EdgeResponse])
async def get_edges(type: str = "", limit: int = 100):
    """Get edges from the knowledge graph."""
    client = get_graph_client()
    if not client:
        return [
            EdgeResponse(source="mock_0", target="mock_1", type=type or "POSTED")
            for _ in range(min(limit, 5))
        ]

    type_filter = f":{type}" if type else ""
    query = f"MATCH (a)-[r{type_filter}]->(b) RETURN a, r, b LIMIT $limit"
    try:
        results = client.execute_read(query, {"limit": limit})
        return [
            EdgeResponse(
                source=str(r["a"].get("name", r["a"].get("id", ""))),
                target=str(r["b"].get("name", r["b"].get("id", ""))),
                type=type(r["r"]).__name__ if hasattr(r["r"], "__name__") else str(type),
                properties={},
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=GraphStatsResponse)
async def get_stats():
    """Get graph statistics."""
    client = get_graph_client()
    if not client:
        return GraphStatsResponse(
            total_nodes=0, total_edges=0,
            node_counts={"Author": 0, "RedditPost": 0},
            edge_counts={"POSTED": 0},
        )

    try:
        node_count = client.execute_read("MATCH (n) RETURN count(n) as c", {})[0]["c"]
        edge_count = client.execute_read("MATCH ()-[r]->() RETURN count(r) as c", {})[0]["c"]
        return GraphStatsResponse(
            total_nodes=node_count,
            total_edges=edge_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communities", response_model=List[CommunityResponse])
async def get_communities():
    """Get detected communities."""
    client = get_graph_client()
    if not client:
        return [
            CommunityResponse(id=i, size=50 - i * 10, modularity_score=0.65)
            for i in range(3)
        ]

    try:
        results = client.execute_read(
            "MATCH (c:Community) RETURN c ORDER BY c.size DESC LIMIT 50", {}
        )
        return [
            CommunityResponse(
                id=r["c"].get("id", 0),
                size=r["c"].get("size", 0),
                modularity_score=r["c"].get("modularity_score", 0.0),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/influence", response_model=List[InfluenceResponse])
async def get_top_influencers(limit: int = 20):
    """Get top influential nodes."""
    client = get_graph_client()
    if not client:
        return [
            InfluenceResponse(node_id=f"user_{i}", score=1.0 - i * 0.05)
            for i in range(min(limit, 10))
        ]

    try:
        results = client.execute_read(
            "MATCH (a:Author) WHERE a.influence_score IS NOT NULL "
            "RETURN a ORDER BY a.influence_score DESC LIMIT $limit",
            {"limit": limit},
        )
        return [
            InfluenceResponse(
                node_id=r["a"].get("name", ""),
                score=r["a"].get("influence_score", 0.0),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
