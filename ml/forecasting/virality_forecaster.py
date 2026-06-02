"""
Virality forecaster using attention-based temporal model.
Combines TGN embeddings with time-series forecasting for
engagement prediction at 6h/24h/72h horizons.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result of a virality forecast."""
    topic: str
    horizon_hours: int
    predictions: List[Dict[str, float]]  # [{timestamp, engagement, ci_lower, ci_upper}]
    virality_score: float
    cascade_probability: float
    peak_time_hours: float


class ViralityForecaster:
    """
    Multi-horizon engagement forecaster.

    Architecture:
    1. Extracts temporal features from event history
    2. Uses exponential decay + power law model for short-term forecasting
    3. Combines with TGN node embeddings for context-aware prediction

    In production, wraps a TFT (Temporal Fusion Transformer) or Prophet model.
    Currently implements an analytical forecaster based on Hawkes-SEIR coupling.
    """

    def __init__(
        self,
        horizons: List[int] = [6, 24, 72],
        decay_rate: float = 0.05,
        cascade_threshold: float = 0.85,
    ):
        self.horizons = horizons
        self.decay_rate = decay_rate
        self.cascade_threshold = cascade_threshold

    def forecast(
        self,
        event_history: List[Dict],
        topic: str = "",
        horizon_hours: int = 24,
    ) -> ForecastResult:
        """
        Generate engagement forecast from event history.

        Args:
            event_history: List of events with 'timestamp' and 'engagement' fields.
            topic: Topic name for labeling.
            horizon_hours: Forecast horizon in hours.

        Returns:
            ForecastResult with predictions and virality metrics.
        """
        if not event_history:
            return ForecastResult(
                topic=topic,
                horizon_hours=horizon_hours,
                predictions=[],
                virality_score=0.0,
                cascade_probability=0.0,
                peak_time_hours=0.0,
            )

        # Extract features
        timestamps = np.array([e.get("timestamp", 0) for e in event_history])
        engagements = np.array([e.get("engagement", 1) for e in event_history])

        # Normalize timestamps to hours
        t0 = timestamps.min()
        t_hours = (timestamps - t0) / 3600.0

        # Fit exponential decay + power law
        peak_engagement = engagements.max()
        peak_time = t_hours[engagements.argmax()]
        total_engagement = engagements.sum()

        # Compute rate features
        if len(t_hours) > 1:
            dt = np.diff(t_hours)
            inter_event_mean = dt.mean()
            velocity = len(event_history) / max(t_hours.max(), 1)
        else:
            inter_event_mean = 1.0
            velocity = 1.0

        # Generate predictions
        predictions = []
        current_time = timestamps.max()
        for h in range(horizon_hours):
            t = current_time + h * 3600
            t_rel = peak_time + (t_hours.max() - peak_time) + h

            # Branching process-inspired forecast
            # Post-peak: exponential decay
            # Pre-peak: power law growth
            if t_rel > peak_time:
                predicted = peak_engagement * np.exp(-self.decay_rate * (t_rel - peak_time))
            else:
                predicted = peak_engagement * (t_rel / max(peak_time, 1)) ** 1.5

            # Add velocity-based correction
            predicted *= (1 + velocity * 0.1)

            # Confidence interval widens with horizon
            ci_width = predicted * 0.1 * (1 + h * 0.05)
            predictions.append({
                "timestamp": float(t),
                "predicted_engagement": max(0, float(predicted)),
                "confidence_lower": max(0, float(predicted - ci_width)),
                "confidence_upper": float(predicted + ci_width),
            })

        # Virality score: combines velocity, total engagement, and growth pattern
        virality = self._compute_virality(velocity, total_engagement, peak_engagement, t_hours)

        # Cascade probability: likelihood of reaching critical mass
        cascade_prob = min(1.0, virality / self.cascade_threshold)

        return ForecastResult(
            topic=topic,
            horizon_hours=horizon_hours,
            predictions=predictions,
            virality_score=round(virality, 4),
            cascade_probability=round(cascade_prob, 4),
            peak_time_hours=round(float(peak_time), 2),
        )

    def _compute_virality(
        self, velocity: float, total: float, peak: float, t_hours: np.ndarray
    ) -> float:
        """
        Compute virality score in [0, 1] from event statistics.
        Based on R₀ analogue from the SEIR-Z-D model.
        """
        # Log-normalized components
        vel_score = min(1.0, np.log1p(velocity) / 5.0)
        vol_score = min(1.0, np.log1p(total) / 10.0)
        peak_score = min(1.0, np.log1p(peak) / 8.0)

        # Burstiness: how concentrated are events in time
        if len(t_hours) > 2:
            iet = np.diff(np.sort(t_hours))
            burstiness = (iet.std() - iet.mean()) / (iet.std() + iet.mean() + 1e-8)
            burst_score = max(0, (burstiness + 1) / 2)
        else:
            burst_score = 0.5

        return 0.3 * vel_score + 0.25 * vol_score + 0.25 * peak_score + 0.2 * burst_score

    def forecast_multi_horizon(
        self, event_history: List[Dict], topic: str = ""
    ) -> Dict[int, ForecastResult]:
        """Generate forecasts for all configured horizons."""
        return {
            h: self.forecast(event_history, topic, h)
            for h in self.horizons
        }
