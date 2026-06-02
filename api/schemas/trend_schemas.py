"""Pydantic schemas for trend/forecast API endpoints."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class TrendItem(BaseModel):
    topic: str
    score: float
    platform: str = ""
    sentiment: str = ""
    velocity: float = 0.0


class TrendResponse(BaseModel):
    timestamp: float
    trends: List[TrendItem]


class ForecastRequest(BaseModel):
    topic: str
    horizon_hours: int = 24
    platforms: List[str] = []


class ForecastPoint(BaseModel):
    timestamp: float
    predicted_engagement: float
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0


class ForecastResponse(BaseModel):
    topic: str
    horizon_hours: int
    predictions: List[ForecastPoint]
    virality_score: float = 0.0
    cascade_threshold: float = 0.0


class NarrativeResponse(BaseModel):
    id: str
    summary: str
    platforms: List[str]
    first_seen: float
    sentiment_avg: float
    virality_score: float
    related_topics: List[str] = []
