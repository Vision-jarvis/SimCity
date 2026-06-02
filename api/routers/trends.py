"""Trend and forecasting API endpoints."""

from fastapi import APIRouter
from typing import List
import time

from api.schemas.trend_schemas import (
    TrendItem, TrendResponse, ForecastRequest, ForecastResponse,
    ForecastPoint, NarrativeResponse,
)

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("/current", response_model=TrendResponse)
async def get_current_trends(limit: int = 20):
    """Get currently trending topics across all platforms."""
    # In production, this queries the graph + NLP pipeline
    # For now, return structured mock data
    mock_trends = [
        TrendItem(topic="AI Regulation", score=0.95, platform="reddit", sentiment="negative", velocity=2.3),
        TrendItem(topic="Climate Summit 2026", score=0.87, platform="gdelt", sentiment="neutral", velocity=1.8),
        TrendItem(topic="Crypto Market Crash", score=0.82, platform="hackernews", sentiment="negative", velocity=3.1),
        TrendItem(topic="Open Source LLMs", score=0.78, platform="hackernews", sentiment="positive", velocity=1.5),
        TrendItem(topic="Election Disinformation", score=0.75, platform="reddit", sentiment="negative", velocity=2.7),
    ]
    return TrendResponse(timestamp=time.time(), trends=mock_trends[:limit])


@router.post("/forecast", response_model=ForecastResponse)
async def forecast_trend(request: ForecastRequest):
    """Forecast engagement trajectory for a topic."""
    # In production, this uses the TFT/Prophet model
    import math
    predictions = []
    now = time.time()
    for h in range(request.horizon_hours):
        t = now + h * 3600
        # Simple decay curve as placeholder
        engagement = 1000 * math.exp(-0.05 * h) * (1 + 0.3 * math.sin(h / 6))
        predictions.append(ForecastPoint(
            timestamp=t,
            predicted_engagement=max(0, engagement),
            confidence_lower=max(0, engagement * 0.7),
            confidence_upper=engagement * 1.3,
        ))

    return ForecastResponse(
        topic=request.topic,
        horizon_hours=request.horizon_hours,
        predictions=predictions,
        virality_score=0.72,
        cascade_threshold=0.85,
    )


@router.get("/narratives", response_model=List[NarrativeResponse])
async def get_narratives(limit: int = 10):
    """Get active narrative threads across platforms."""
    mock_narratives = [
        NarrativeResponse(
            id="narr_001",
            summary="AI safety debate intensifies as new models emerge",
            platforms=["reddit", "hackernews", "gdelt"],
            first_seen=time.time() - 86400 * 3,
            sentiment_avg=-0.15,
            virality_score=0.88,
            related_topics=["AI Regulation", "Open Source LLMs"],
        ),
        NarrativeResponse(
            id="narr_002",
            summary="Cryptocurrency market volatility triggers cross-platform discussion",
            platforms=["reddit", "youtube"],
            first_seen=time.time() - 86400,
            sentiment_avg=-0.42,
            virality_score=0.73,
            related_topics=["Crypto Market Crash", "DeFi"],
        ),
    ]
    return mock_narratives[:limit]
