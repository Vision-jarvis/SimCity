"""
Model evaluation utilities for the SimCity training pipeline.

These helpers are intentionally framework-light (NumPy only) so they can run
in CI without PyTorch Geometric installed. They turn raw predictions and a
trained model's reported metrics into an :class:`EvaluationReport` that the
training pipeline logs to MLflow and the scheduler uses to gate promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Standard regression metrics for virality/engagement targets.

    Ignores NaN targets (the synthetic dataset uses NaN for events whose
    future-engagement window falls outside the observation horizon).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    yt, yp = yt[mask], yp[mask]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(mask.sum())}


def classification_metrics(
    y_true: Sequence[float], y_score: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    """Precision/recall/F1 for the binary virality classifier."""
    yt = np.asarray(y_true, dtype=float)
    yp = (np.asarray(y_score, dtype=float) >= threshold).astype(float)
    tp = float(np.sum((yp == 1) & (yt == 1)))
    fp = float(np.sum((yp == 1) & (yt == 0)))
    fn = float(np.sum((yp == 0) & (yt == 1)))
    tn = float(np.sum((yp == 0) & (yt == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(len(yt), 1)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


@dataclass
class EvaluationReport:
    """Aggregated evaluation result for one trained model version."""

    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    primary_metric: str = "virality_mae"
    primary_value: float = float("inf")
    higher_is_better: bool = False
    n_samples: int = 0

    def is_better_than(self, other: Optional["EvaluationReport"], min_delta: float = 0.0) -> bool:
        """Whether this report beats ``other`` on the primary metric."""
        if other is None:
            return True
        if self.higher_is_better:
            return self.primary_value > other.primary_value + min_delta
        return self.primary_value < other.primary_value - min_delta

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelEvaluator:
    """Builds an :class:`EvaluationReport` from predictions or raw metrics."""

    def __init__(
        self,
        model_name: str,
        primary_metric: str = "virality_mae",
        higher_is_better: bool = False,
    ):
        self.model_name = model_name
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better

    def from_predictions(
        self, y_true: Sequence[float], y_pred: Sequence[float]
    ) -> EvaluationReport:
        reg = regression_metrics(y_true, y_pred)
        metrics = {f"virality_{k}": v for k, v in reg.items()}
        return self._build(metrics, n=reg.get("n", 0))

    def from_metrics(self, metrics: Dict[str, float], n_samples: int = 0) -> EvaluationReport:
        return self._build(dict(metrics), n=n_samples)

    def _build(self, metrics: Dict[str, float], n: int) -> EvaluationReport:
        primary = metrics.get(self.primary_metric)
        if primary is None:
            # Fall back to the first metric so the report is always orderable.
            primary = next(iter(metrics.values()), float("inf"))
            logger.debug(
                "Primary metric %s missing; using fallback %s",
                self.primary_metric,
                primary,
            )
        return EvaluationReport(
            model_name=self.model_name,
            metrics=metrics,
            primary_metric=self.primary_metric,
            primary_value=float(primary),
            higher_is_better=self.higher_is_better,
            n_samples=n,
        )
