"""Pydantic schemas for graph API endpoints."""

from pydantic import BaseModel
from typing import List, Dict, Any


class NodeResponse(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = {}


class EdgeResponse(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class GraphStatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    node_counts: Dict[str, int] = {}
    edge_counts: Dict[str, int] = {}


class CommunityResponse(BaseModel):
    id: int
    size: int
    modularity_score: float
    members: List[str] = []


class InfluenceResponse(BaseModel):
    node_id: str
    score: float
    pagerank: float = 0.0
    degree: int = 0


class GraphQueryRequest(BaseModel):
    cypher: str
    parameters: Dict[str, Any] = {}
    limit: int = 100
