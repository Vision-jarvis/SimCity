# MLOps Lifecycle

SimCity's training lifecycle is built so it runs **anywhere**, with a
self-hosted MLflow server when available, and a transparent local JSON store
when not (CI, offline, no-server dev).

## Components

| Component | File | Role |
|---|---|---|
| `ExperimentTracker` | `ml/training/pipeline.py` | Log params/metrics/artifacts (MLflow or local) |
| `TrainingPipeline` | `ml/training/pipeline.py` | Track → evaluate → register → auto-promote |
| `ModelRegistry` | `ml/registry/model_registry.py` | Versioning + stage transitions |
| `ModelEvaluator` | `ml/training/evaluate.py` | Comparable `EvaluationReport`s |
| `RetrainingScheduler` | `ml/training/scheduler.py` | Drift + staleness gating |

## Lifecycle

```
1. TrainingPipeline.run(train_fn)
   ├─ tracker.start_run()            # MLflow run or local-<ts>.json
   ├─ tracker.log_params(hparams)
   ├─ metrics, ckpt = train_fn()     # your training code
   ├─ tracker.log_metrics(metrics)
   ├─ registry.register(...)         # new version
   └─ if report.is_better_than(prod): registry.promote(..., "Production")

2. RetrainingScheduler.run(train_fn, observed_metrics)
   ├─ check_drift()                  # degradation OR staleness
   └─ if drift.should_retrain: TrainingPipeline.run(...)
```

## Promotion gate

A new version is promoted to `Production` only when it beats the incumbent on
the primary metric (default `virality_mae`, lower-is-better) by at least
`promote_min_delta`. Otherwise the existing production model is retained.

## Drift gate

`RetrainingScheduler.check_drift()` returns `should_retrain=True` when:
- there is **no** production model yet, or
- the model is **stale** (`age >= max_staleness_days`, default 7), or
- the observed primary metric **degraded** by `>= degradation_threshold`
  (default 10% relative).

## CLI

```bash
# Tracked training pass (use --demo to skip PyTorch Geometric):
python run_training_pipeline.py train --demo

# Drift check + conditional retrain (nightly CI calls this):
python run_training_pipeline.py schedule --observed-mae 0.55 --demo
python run_training_pipeline.py schedule --force --demo   # force a retrain
```

## Connecting a real MLflow server

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT=simcity-internet-twin
mlflow server --host 0.0.0.0 --port 5000   # self-hosted, free
```

With those set, `ExperimentTracker`/`ModelRegistry` automatically use MLflow
instead of the local store, no code changes.

## Wiring real training metrics

`train.py` currently trains and prints; `default_train_fn` in
`ml/training/pipeline.py` returns a placeholder. To log real numbers, refactor
`train.train()` to **return** its final validation/test metrics and a
checkpoint path, then point `default_train_fn` at it. Everything downstream
(tracking, registry, promotion, drift) already consumes that contract.
