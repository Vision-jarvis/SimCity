"""
MLflow-backed model registry for SimCity.

Wraps the MLflow Model Registry to give the rest of the codebase a small,
stable surface for registering trained checkpoints, promoting them through
stages (Staging -> Production), and resolving the current production model.

MLflow is an optional dependency. When it is not installed (or no tracking
server is reachable), this module transparently falls back to a local
JSON-backed registry under ``mlruns_local/`` so that the training pipeline,
scheduler, and tests still work in a zero-dependency environment. This mirrors
the "zero-cost, self-hosted" philosophy of the project: MLflow is self-hosted
when available, and degrades gracefully when it is not.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
LOCAL_REGISTRY_ROOT = Path(os.getenv("SIMCITY_LOCAL_REGISTRY", "mlruns_local"))

# Stage names mirror MLflow's canonical stages.
STAGE_NONE = "None"
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"
VALID_STAGES = {STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED}


@dataclass
class RegisteredModel:
    """A single registered model version."""

    name: str
    version: int
    stage: str
    run_id: str
    artifact_path: str
    metrics: Dict[str, float]
    created_at: float

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelRegistry:
    """
    Thin façade over the MLflow Model Registry with a local fallback.

    Usage:
        registry = ModelRegistry()
        version = registry.register(
            name="simcity-tgn",
            artifact_path="checkpoints/tgn_epoch15.pt",
            run_id="abc123",
            metrics={"virality_mae": 0.42, "hawkes_nll": 1.13},
        )
        registry.promote("simcity-tgn", version, stage="Production")
        prod = registry.get_production_model("simcity-tgn")
    """

    def __init__(self, tracking_uri: str = DEFAULT_TRACKING_URI):
        self.tracking_uri = tracking_uri
        self._mlflow = None
        self._client = None
        self._use_mlflow = self._try_init_mlflow()
        if not self._use_mlflow:
            LOCAL_REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "MLflow unavailable; using local JSON registry at %s",
                LOCAL_REGISTRY_ROOT.resolve(),
            )

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #
    def _try_init_mlflow(self) -> bool:
        try:
            # Fail fast instead of minutes of HTTP retries when the tracking
            # server is unreachable (e.g. CI has no MLflow server running).
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient(tracking_uri=self.tracking_uri)

            # MLflow 3.x removed registry stages (transition_model_version_stage);
            # this registry's promote/list logic depends on them.
            if not hasattr(client, "transition_model_version_stage"):
                logger.warning(
                    "MLflow %s lacks registry stages (removed in 3.x); "
                    "falling back to local registry",
                    getattr(mlflow, "__version__", "?"),
                )
                return False

            # Reachability probe: client construction is lazy, so make one
            # cheap real call before committing to the MLflow backend.
            client.search_registered_models(max_results=1)

            self._mlflow = mlflow
            self._client = client
            logger.info("ModelRegistry using MLflow at %s", self.tracking_uri)
            return True
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning(
                "MLflow unavailable at %s (%s); falling back to local registry",
                self.tracking_uri, exc.__class__.__name__,
            )
            return False

    @property
    def backend(self) -> str:
        return "mlflow" if self._use_mlflow else "local"

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(
        self,
        name: str,
        artifact_path: str,
        run_id: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> int:
        """Register a new model version and return its integer version."""
        metrics = metrics or {}
        if self._use_mlflow:
            return self._register_mlflow(name, artifact_path, run_id, metrics)
        return self._register_local(name, artifact_path, run_id, metrics)

    def _register_mlflow(self, name, artifact_path, run_id, metrics) -> int:
        # Ensure the registered model exists.
        try:
            self._client.create_registered_model(name)
        except Exception:
            pass  # already exists
        model_uri = f"runs:/{run_id}/model" if run_id else artifact_path
        mv = self._client.create_model_version(
            name=name, source=model_uri, run_id=run_id or None
        )
        return int(mv.version)

    def _register_local(self, name, artifact_path, run_id, metrics) -> int:
        index = self._load_local_index(name)
        version = len(index["versions"]) + 1
        # Copy the artifact into the registry store if it is a real file.
        stored_path = artifact_path
        src = Path(artifact_path) if artifact_path else None
        if src is not None and src.is_file():
            dst_dir = LOCAL_REGISTRY_ROOT / name / f"v{version}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            stored_path = str(dst)
        record = RegisteredModel(
            name=name,
            version=version,
            stage=STAGE_NONE,
            run_id=run_id or f"local-{int(time.time())}",
            artifact_path=stored_path,
            metrics=metrics,
            created_at=time.time(),
        )
        index["versions"].append(record.to_dict())
        self._save_local_index(name, index)
        logger.info("Registered %s v%d (local)", name, version)
        return version

    # ------------------------------------------------------------------ #
    # Promotion
    # ------------------------------------------------------------------ #
    def promote(self, name: str, version: int, stage: str = STAGE_PRODUCTION) -> None:
        """Transition a model version to a stage, archiving prior occupants."""
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {VALID_STAGES}")
        if self._use_mlflow:
            self._client.transition_model_version_stage(
                name=name,
                version=str(version),
                stage=stage,
                archive_existing_versions=(stage == STAGE_PRODUCTION),
            )
            return
        index = self._load_local_index(name)
        for v in index["versions"]:
            if stage == STAGE_PRODUCTION and v["stage"] == STAGE_PRODUCTION:
                v["stage"] = STAGE_ARCHIVED
            if v["version"] == version:
                v["stage"] = stage
        self._save_local_index(name, index)
        logger.info("Promoted %s v%d -> %s (local)", name, version, stage)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def list_versions(self, name: str) -> List[RegisteredModel]:
        if self._use_mlflow:
            versions = self._client.search_model_versions(f"name='{name}'")
            return [
                RegisteredModel(
                    name=name,
                    version=int(v.version),
                    stage=v.current_stage,
                    run_id=v.run_id or "",
                    artifact_path=v.source,
                    metrics={},
                    created_at=float(v.creation_timestamp or 0) / 1000.0,
                )
                for v in versions
            ]
        index = self._load_local_index(name)
        return [RegisteredModel(**v) for v in index["versions"]]

    def get_production_model(self, name: str) -> Optional[RegisteredModel]:
        """Return the version currently in Production, or None."""
        for v in self.list_versions(name):
            if v.stage == STAGE_PRODUCTION:
                return v
        return None

    def get_latest_model(self, name: str) -> Optional[RegisteredModel]:
        versions = self.list_versions(name)
        if not versions:
            return None
        return max(versions, key=lambda v: v.version)

    # ------------------------------------------------------------------ #
    # Local JSON index helpers
    # ------------------------------------------------------------------ #
    def _index_path(self, name: str) -> Path:
        return LOCAL_REGISTRY_ROOT / name / "index.json"

    def _load_local_index(self, name: str) -> Dict:
        path = self._index_path(name)
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"name": name, "versions": []}

    def _save_local_index(self, name: str, index: Dict) -> None:
        path = self._index_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)
