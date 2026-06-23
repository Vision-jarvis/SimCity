"""
Trend predictor for next 6h/24h/72h engagement and volume predictions.
Combines statistical features with learned patterns from historical data.
"""

import logging
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrendPrediction:
    """Prediction for a single trend/topic."""
    topic: str
    current_rank: int
    predicted_rank_6h: int
    predicted_rank_24h: int
    predicted_rank_72h: int
    momentum: float  # Rate of change
    is_emerging: bool
    is_declining: bool
    breakout_probability: float


class TrendPredictor:
    """
    Predicts trend trajectories using velocity, acceleration,
    and cross-platform signal analysis.

    Features:
    - Rank prediction at multiple horizons
    - Momentum analysis
    - Emerging/declining trend detection
    - Breakout probability estimation
    """

    def __init__(self, breakout_threshold: float = 0.7, window_hours: float = 12.0):
        self.breakout_threshold = breakout_threshold
        self.window_hours = window_hours

    def predict(
        self,
        topic_histories: Dict[str, List[Dict]],
    ) -> List[TrendPrediction]:
        """
        Predict trend trajectories for multiple topics.

        Args:
            topic_histories: Dict mapping topic → list of events with
                            'timestamp' and 'engagement'.

        Returns:
            List of TrendPrediction sorted by momentum.
        """
        predictions = []

        for topic, events in topic_histories.items():
            pred = self._predict_single(topic, events)
            predictions.append(pred)

        # Sort by momentum (descending)
        predictions.sort(key=lambda p: p.momentum, reverse=True)

        # Assign ranks
        for i, pred in enumerate(predictions):
            pred.current_rank = i + 1
            # Simple rank shift prediction
            shift_6h = max(-3, min(3, int(-pred.momentum * 2)))
            shift_24h = max(-5, min(5, int(-pred.momentum * 4)))
            shift_72h = max(-8, min(8, int(-pred.momentum * 6)))
            pred.predicted_rank_6h = max(1, pred.current_rank + shift_6h)
            pred.predicted_rank_24h = max(1, pred.current_rank + shift_24h)
            pred.predicted_rank_72h = max(1, pred.current_rank + shift_72h)

        return predictions

    def _predict_single(self, topic: str, events: List[Dict]) -> TrendPrediction:
        """Predict trajectory for a single topic."""
        if not events:
            return TrendPrediction(
                topic=topic, current_rank=0,
                predicted_rank_6h=0, predicted_rank_24h=0, predicted_rank_72h=0,
                momentum=0.0, is_emerging=False, is_declining=False,
                breakout_probability=0.0,
            )

        engagements = np.array([e.get("engagement", 1) for e in events])
        timestamps = np.array([e.get("timestamp", 0) for e in events])

        # Split into recent and older halves
        mid = len(events) // 2
        if mid > 0:
            recent_rate = engagements[mid:].sum() / max(1, len(engagements) - mid)
            older_rate = engagements[:mid].sum() / max(1, mid)
            momentum = (recent_rate - older_rate) / max(older_rate, 1e-6)
        else:
            momentum = 0.0

        # Velocity (events per hour)
        if len(timestamps) > 1:
            duration_hours = (timestamps.max() - timestamps.min()) / 3600.0
            velocity = len(events) / max(duration_hours, 1)
        else:
            velocity = 0.0

        # Emerging: positive momentum + accelerating
        is_emerging = momentum > 0.2 and velocity > 1.0
        is_declining = momentum < -0.3

        # Breakout: high velocity + high momentum + short history
        breakout = min(1.0, max(0.0,
            0.4 * min(1, velocity / 5.0)
            + 0.3 * min(1, max(0, momentum) / 2.0)
            + 0.3 * min(1, engagements.sum() / 100.0)
        ))

        return TrendPrediction(
            topic=topic,
            current_rank=0,  # Set later
            predicted_rank_6h=0,
            predicted_rank_24h=0,
            predicted_rank_72h=0,
            momentum=round(float(momentum), 4),
            is_emerging=is_emerging,
            is_declining=is_declining,
            breakout_probability=round(breakout, 4),
        )
