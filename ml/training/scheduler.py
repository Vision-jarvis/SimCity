"""
Scheduled retraining + drift detection for SimCity models.

Designed to run from the nightly GitHub Actions workflow (see
``.github/workflows/`` and the project roadmap, Phase 3). On each invocation it:

1. Loads the production model's reference metrics from the registry.
2. Compares them against freshly observed metrics on recent data.
3. Decides whether drift warrants a retrain (degradation beyond a threshold,
   or a maximum staleness interval elapsed).
4. Optionally triggers :class:`ml.training.TrainingPipeline`.

The drift test is deliberately simple and dependency-free (a relative-change
check on the primary metric plus a staleness clock) so it runs anywhere.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

from ml.registry.model_registry import ModelRegistry
from ml.training.evaluate import EvaluationReport
from ml.training.pipeline import TrainingPipeline, TrainFn

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400.0


@dataclass
class DriftReport:
    """Outcome of a drift check."""

    should_retrain: bool
    reason: str
    primary_metric: str
    reference_value: Optional[float]
    observed_value: Optional[float]
    relative_change: Optional[float]
    staleness_days: Optional[float]

    def to_dict(self) -> Dict:
        return asdict(self)


class RetrainingScheduler:
    """Decides when to retrain and (optionally) drives the training pipeline."""

    def __init__(
        self,
        model_name: str = "simcity-tgn",
        primary_metric: str = "virality_mae",
        higher_is_better: bool = False,
        degradation_threshold: float = 0.10,  # 10% relative degradation
        max_staleness_days: float = 7.0,
        registry: Optional[ModelRegistry] = None,
    ):
        self.model_name = model_name
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        self.degradation_threshold = degradation_threshold
        self.max_staleness_days = max_staleness_days
        self.registry = registry or ModelRegistry()

    # ------------------------------------------------------------------ #
    def check_drift(self, observed_metrics: Optional[Dict[str, float]] = None) -> DriftReport:
        """Compare observed metrics against the production reference."""
        prod = self.registry.get_production_model(self.model_name)

        if prod is None:
            return DriftReport(
                should_retrain=True,
                reason="no_production_model",
                primary_metric=self.primary_metric,
                reference_value=None,
                observed_value=None,
                relative_change=None,
                staleness_days=None,
            )

        staleness_days = (time.time() - prod.created_at) / SECONDS_PER_DAY
        ref = prod.metrics.get(self.primary_metric)
        obs = (observed_metrics or {}).get(self.primary_metric)

        # Staleness gate.
        if staleness_days >= self.max_staleness_days:
            return DriftReport(
                should_retrain=True,
                reason=f"stale_model ({staleness_days:.1f}d >= {self.max_staleness_days}d)",
                primary_metric=self.primary_metric,
                reference_value=ref,
                observed_value=obs,
                relative_change=None,
                staleness_days=staleness_days,
            )

        # Performance-degradation gate.
        if ref is not None and obs is not None and abs(ref) > 1e-12:
            rel = (obs - ref) / abs(ref)
            # For "lower is better", positive rel == degradation; flip otherwise.
            degradation = rel if not self.higher_is_better else -rel
            if degradation >= self.degradation_threshold:
                return DriftReport(
                    should_retrain=True,
                    reason=f"performance_drift ({degradation:+.1%})",
                    primary_metric=self.primary_metric,
                    reference_value=ref,
                    observed_value=obs,
                    relative_change=rel,
                    staleness_days=staleness_days,
                )
            return DriftReport(
                should_retrain=False,
                reason=f"within_tolerance ({degradation:+.1%})",
                primary_metric=self.primary_metric,
                reference_value=ref,
                observed_value=obs,
                relative_change=rel,
                staleness_days=staleness_days,
            )

        return DriftReport(
            should_retrain=False,
            reason="no_observed_metrics",
            primary_metric=self.primary_metric,
            reference_value=ref,
            observed_value=obs,
            relative_change=None,
            staleness_days=staleness_days,
        )

    # ------------------------------------------------------------------ #
    def run(
        self,
        train_fn: TrainFn,
        observed_metrics: Optional[Dict[str, float]] = None,
        hyperparams: Optional[Dict] = None,
        force: bool = False,
    ) -> Tuple[DriftReport, Optional[EvaluationReport]]:
        """Check drift and retrain if warranted.

        Returns ``(drift_report, evaluation_report_or_None)``.
        """
        drift = self.check_drift(observed_metrics)
        if not (drift.should_retrain or force):
            logger.info("No retrain needed: %s", drift.reason)
            return drift, None

        logger.info("Triggering retraining: %s", drift.reason if not force else "forced")
        pipeline = TrainingPipeline(
            model_name=self.model_name,
            primary_metric=self.primary_metric,
            higher_is_better=self.higher_is_better,
            registry=self.registry,
        )
        report = pipeline.run(train_fn, hyperparams=hyperparams)
        return drift, report
