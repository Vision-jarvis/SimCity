"""
End-to-end training pipeline with MLflow experiment tracking.

Responsibilities:
1. Open an MLflow run (or a local JSON run when MLflow is unavailable).
2. Log hyperparameters.
3. Execute a training function that returns final metrics + an optional
   checkpoint path.
4. Evaluate the run and, if it beats the current Production model, register
   and promote the new version via :class:`ml.registry.ModelRegistry`.

The training function is injectable so the pipeline stays usable in CI without
PyTorch Geometric. By default it adapts the project's ``train.train`` entry
point (which requires PyG). See ``run_training_pipeline.py`` for a CLI.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from ml.registry.model_registry import ModelRegistry
from ml.training.evaluate import ModelEvaluator, EvaluationReport

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "simcity-internet-twin")
LOCAL_RUNS_ROOT = Path(os.getenv("SIMCITY_LOCAL_RUNS", "mlruns_local/runs"))

# A training function returns (metrics, checkpoint_path).
TrainFn = Callable[[Dict], Tuple[Dict[str, float], Optional[str]]]


class ExperimentTracker:
    """MLflow tracker with a local JSON fallback.

    Mirrors the small subset of the MLflow fluent API we actually use so call
    sites read the same regardless of backend.
    """

    def __init__(self, experiment: str = DEFAULT_EXPERIMENT, tracking_uri: Optional[str] = None):
        self.experiment = experiment
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self._mlflow = None
        self.run_id: Optional[str] = None
        self._local_record: Dict = {}
        self._use_mlflow = self._try_init()

    def _try_init(self) -> bool:
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            self._mlflow = mlflow
            logger.info("ExperimentTracker using MLflow at %s", self.tracking_uri)
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("MLflow unavailable (%s); logging runs locally", exc)
            LOCAL_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
            return False

    @property
    def backend(self) -> str:
        return "mlflow" if self._use_mlflow else "local"

    @contextmanager
    def start_run(self, run_name: str = ""):
        if self._use_mlflow:
            with self._mlflow.start_run(run_name=run_name or None) as run:
                self.run_id = run.info.run_id
                yield self
        else:
            self.run_id = f"local-{int(time.time())}"
            self._local_record = {
                "run_id": self.run_id,
                "run_name": run_name,
                "experiment": self.experiment,
                "params": {},
                "metrics": {},
                "artifacts": [],
                "started_at": time.time(),
            }
            try:
                yield self
            finally:
                path = LOCAL_RUNS_ROOT / f"{self.run_id}.json"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(self._local_record, fh, indent=2)
                logger.info("Local run logged to %s", path)

    def log_params(self, params: Dict) -> None:
        clean = {k: v for k, v in params.items() if v is not None}
        if self._use_mlflow:
            self._mlflow.log_params(clean)
        else:
            self._local_record["params"].update({k: str(v) for k, v in clean.items()})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items() if _is_number(v)}
        if self._use_mlflow:
            self._mlflow.log_metrics(numeric, step=step)
        else:
            self._local_record["metrics"].update(numeric)

    def log_artifact(self, path: str) -> None:
        if not path or not Path(path).exists():
            return
        if self._use_mlflow:
            self._mlflow.log_artifact(path)
        else:
            self._local_record["artifacts"].append(path)


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class TrainingPipeline:
    """Orchestrates a tracked training run with automatic model promotion."""

    def __init__(
        self,
        model_name: str = "simcity-tgn",
        experiment: str = DEFAULT_EXPERIMENT,
        primary_metric: str = "virality_mae",
        higher_is_better: bool = False,
        promote_min_delta: float = 0.0,
        registry: Optional[ModelRegistry] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.model_name = model_name
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        self.promote_min_delta = promote_min_delta
        self.registry = registry or ModelRegistry()
        self.tracker = tracker or ExperimentTracker(experiment=experiment)
        self.evaluator = ModelEvaluator(
            model_name, primary_metric=primary_metric, higher_is_better=higher_is_better
        )

    def run(
        self,
        train_fn: TrainFn,
        hyperparams: Optional[Dict] = None,
        run_name: str = "",
        auto_promote: bool = True,
    ) -> EvaluationReport:
        """Execute one tracked training run.

        Returns the :class:`EvaluationReport` for the run.
        """
        hyperparams = hyperparams or {}
        run_name = run_name or f"{self.model_name}-{int(time.time())}"

        with self.tracker.start_run(run_name=run_name):
            self.tracker.log_params(hyperparams)
            logger.info("Starting training run '%s' (backend=%s)", run_name, self.tracker.backend)

            metrics, checkpoint = train_fn(hyperparams)

            self.tracker.log_metrics(metrics)
            if checkpoint:
                self.tracker.log_artifact(checkpoint)

            report = self.evaluator.from_metrics(metrics, n_samples=int(metrics.get("n", 0)))

            version = self.registry.register(
                name=self.model_name,
                artifact_path=checkpoint or "",
                run_id=self.tracker.run_id or "",
                metrics=report.metrics,
            )

            if auto_promote:
                self._maybe_promote(version, report)

        return report

    def _maybe_promote(self, version: int, report: EvaluationReport) -> None:
        current = self.registry.get_production_model(self.model_name)
        current_report = None
        if current is not None:
            current_report = self.evaluator.from_metrics(current.metrics)
        if report.is_better_than(current_report, min_delta=self.promote_min_delta):
            self.registry.promote(self.model_name, version)
            logger.info(
                "Promoted %s v%d to Production (%s=%.4f)",
                self.model_name,
                version,
                self.primary_metric,
                report.primary_value,
            )
        else:
            logger.info(
                "Kept existing Production model; v%d (%s=%.4f) did not beat it",
                version,
                self.primary_metric,
                report.primary_value,
            )


def default_train_fn(hyperparams: Dict) -> Tuple[Dict[str, float], Optional[str]]:
    """Adapter around the project's ``train.train`` entry point.

    Requires PyTorch Geometric. Returns the final validation metrics and a
    checkpoint path. Raises ImportError with a clear message if PyG is missing.
    """
    try:
        import train as train_module  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "default_train_fn requires PyTorch Geometric (torch_geometric). "
            "Install it, or pass a custom train_fn to TrainingPipeline.run()."
        ) from exc
    # train.train() currently trains and prints; wrap to capture metrics here
    # once train.py is refactored to return them. For now, run and report a
    # placeholder so the pipeline contract holds.
    train_module.train()
    return {"virality_mae": float("nan")}, None
