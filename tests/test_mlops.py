"""Tests for the MLOps layer: registry, evaluation, pipeline, scheduler.

These run with the local JSON fallback (no MLflow server required) by pointing
the registry and run store at a temporary directory.
"""

import importlib

import pytest


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Force the local-JSON backend into a temp dir for hermetic tests."""
    monkeypatch.setenv("SIMCITY_LOCAL_REGISTRY", str(tmp_path / "registry"))
    monkeypatch.setenv("SIMCITY_LOCAL_RUNS", str(tmp_path / "runs"))
    # Reload modules so the new env-driven paths take effect.
    import ml.registry.model_registry as reg_mod
    import ml.training.pipeline as pipe_mod
    importlib.reload(reg_mod)
    importlib.reload(pipe_mod)
    reg = reg_mod.ModelRegistry()
    # Guarantee we are exercising the local path, not a live MLflow server.
    reg._use_mlflow = False
    return reg_mod, reg


def test_regression_metrics_ignores_nan():
    from ml.training.evaluate import regression_metrics
    import math

    m = regression_metrics([1.0, 2.0, math.nan, 4.0], [1.0, 2.0, 99.0, 4.0])
    assert m["mae"] == 0.0
    assert m["n"] == 3


def test_evaluation_report_ordering():
    from ml.training.evaluate import ModelEvaluator

    ev = ModelEvaluator("m", primary_metric="virality_mae", higher_is_better=False)
    better = ev.from_metrics({"virality_mae": 0.3})
    worse = ev.from_metrics({"virality_mae": 0.5})
    assert better.is_better_than(worse)
    assert not worse.is_better_than(better)
    assert better.is_better_than(None)  # nothing in production yet


def test_registry_register_and_promote(isolated_registry):
    _, reg = isolated_registry
    assert reg.backend == "local"

    v1 = reg.register("simcity-tgn", artifact_path="", metrics={"virality_mae": 0.5})
    v2 = reg.register("simcity-tgn", artifact_path="", metrics={"virality_mae": 0.4})
    assert (v1, v2) == (1, 2)

    reg.promote("simcity-tgn", v1)
    assert reg.get_production_model("simcity-tgn").version == 1

    # Promoting v2 should archive v1.
    reg.promote("simcity-tgn", v2)
    prod = reg.get_production_model("simcity-tgn")
    assert prod.version == 2
    versions = {x.version: x.stage for x in reg.list_versions("simcity-tgn")}
    assert versions[1] == "Archived"
    assert versions[2] == "Production"


def test_pipeline_auto_promotes_better_model(isolated_registry):
    _, reg = isolated_registry
    from ml.training.pipeline import TrainingPipeline
    pipe = TrainingPipeline(model_name="simcity-tgn", registry=reg)
    pipe.tracker._use_mlflow = False  # force local run store

    def train_fn(_hp):
        return {"virality_mae": 0.40, "n": 100}, None

    report = pipe.run(train_fn, hyperparams={"lr": 1e-3})
    assert report.primary_value == 0.40
    assert reg.get_production_model("simcity-tgn").version == 1


def test_scheduler_triggers_on_staleness(isolated_registry):
    from ml.training.scheduler import RetrainingScheduler

    reg_mod, reg = isolated_registry
    sched = RetrainingScheduler(model_name="simcity-tgn", registry=reg)

    # No production model -> must retrain.
    drift = sched.check_drift()
    assert drift.should_retrain and drift.reason == "no_production_model"

    # Register + promote a fresh model -> within tolerance, no retrain.
    v = reg.register("simcity-tgn", artifact_path="", metrics={"virality_mae": 0.5})
    reg.promote("simcity-tgn", v)
    drift = sched.check_drift({"virality_mae": 0.51})
    assert not drift.should_retrain

    # Big degradation -> retrain.
    drift = sched.check_drift({"virality_mae": 0.70})
    assert drift.should_retrain and "performance_drift" in drift.reason
