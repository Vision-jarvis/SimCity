"""
CLI for the MLflow-tracked training pipeline and the retraining scheduler.

Examples
--------
Run a tracked training pass and auto-promote if it beats production::

    python run_training_pipeline.py train

Run the nightly drift check (used by GitHub Actions) and retrain on drift::

    python run_training_pipeline.py schedule --observed-mae 0.55

When PyTorch Geometric is not installed, pass ``--demo`` to exercise the
tracking + registry + promotion machinery with a synthetic training function.
"""

import argparse
import logging
import random
from typing import Dict, Optional, Tuple

from ml.training.pipeline import TrainingPipeline, default_train_fn
from ml.training.scheduler import RetrainingScheduler

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def demo_train_fn(hyperparams: Dict) -> Tuple[Dict[str, float], Optional[str]]:
    """A stand-in trainer that emits plausible metrics (no PyG required)."""
    base = hyperparams.get("target_mae", 0.5)
    mae = max(0.05, base + random.uniform(-0.1, 0.05))
    return (
        {
            "virality_mae": round(mae, 4),
            "virality_rmse": round(mae * 1.3, 4),
            "hawkes_nll": round(1.0 + random.uniform(-0.2, 0.2), 4),
            "n": 2000,
        },
        None,
    )


def main():
    parser = argparse.ArgumentParser(description="SimCity training pipeline / scheduler")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run a tracked training pass")
    p_train.add_argument("--model-name", default="simcity-tgn")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--demo", action="store_true", help="Use synthetic trainer (no PyG)")

    p_sched = sub.add_parser("schedule", help="Drift check + conditional retrain")
    p_sched.add_argument("--model-name", default="simcity-tgn")
    p_sched.add_argument("--observed-mae", type=float, default=None)
    p_sched.add_argument("--max-staleness-days", type=float, default=7.0)
    p_sched.add_argument("--force", action="store_true")
    p_sched.add_argument("--demo", action="store_true", help="Use synthetic trainer (no PyG)")

    args = parser.parse_args()
    train_fn = demo_train_fn if getattr(args, "demo", False) else default_train_fn

    if args.command == "train":
        pipeline = TrainingPipeline(model_name=args.model_name)
        report = pipeline.run(
            train_fn,
            hyperparams={"lr": args.lr, "epochs": args.epochs},
        )
        print(f"\nRun complete. backend={pipeline.tracker.backend}")
        print(f"Metrics: {report.metrics}")

    elif args.command == "schedule":
        scheduler = RetrainingScheduler(
            model_name=args.model_name,
            max_staleness_days=args.max_staleness_days,
        )
        observed = {"virality_mae": args.observed_mae} if args.observed_mae is not None else None
        drift, report = scheduler.run(train_fn, observed_metrics=observed, force=args.force)
        print(f"\nDrift: should_retrain={drift.should_retrain} reason={drift.reason}")
        if report is not None:
            print(f"Retrained. New metrics: {report.metrics}")


if __name__ == "__main__":
    main()
