# SimCity: Internet Twin Research Prototype

SimCity is a research prototype for an AI digital twin of internet dynamics. The full product vision lives in [internet twin.md](./internet%20twin.md): a zero-cost, real-time system for modeling influence cascades, narrative spread, virality, and platform-level collective behavior.

This repository currently implements the modeling core of that vision: a temporal graph neural network with platform-aware virality prediction over synthetic multiplex internet events.

## What Is Here

- `data/synthetic_generator.py` creates synthetic Reddit, Hacker News, and GDELT-style cascade events with future engagement targets.
- `data/dataset.py` loads events into PyTorch Geometric `TemporalData`, appends platform one-hot and GDELT volume features, and performs chronological train/validation/test splits.
- `models/tgn_core.py` implements the Temporal Graph Network memory and temporal graph-attention embedding stack.
- `models/virality_head.py` predicts SEIR-style infectivity, platform Hawkes parameters, and log-engagement virality.
- `models/loss.py` combines TGN contrastive learning, Hawkes likelihood, and virality regression with homoscedastic uncertainty weighting.
- `train.py`, `evaluate.py`, `dry_run.py`, and `run_diagnostics.py` provide training, validation, smoke testing, and gradient diagnostics.
- `baselines/` contains static GNN and multivariate Hawkes baselines for ablation work.
- `run_hawkes_baseline.py` trains a CPU-friendly static Hawkes benchmark without requiring PyTorch Geometric.
- `analysis_tools/cascade_monitor.py` converts Hawkes residuals into event-level cascade alert scores.
- `run_cascade_monitor.py` trains the static Hawkes baseline and emits anomaly summaries for held-out stream events.
- `main.tex` contains the research writeup draft.

## Features Implemented

I implemented the previously missing neural Hawkes cascade objective from the Internet Twin roadmap, then extended it with streaming cascade memory across temporal mini-batches.

Before this change, the virality head produced Hawkes-style `alpha` and `gamma` parameters, but training set Hawkes NLL to `0.0`. Those parameters were not identifiable and the model was not actually learning cross-platform temporal excitation.

The model now has:

- A differentiable `NeuralHawkesLoss` in `models/hawkes.py`.
- A `StreamingHawkesLoss` with bounded P x P platform excitation state for long chronological streams.
- Event-level exogenous background intensity `mu` predicted from narrative embedding plus GDELT volume.
- Cross-platform excitation intensity:
  `lambda_i(t_k) = mu_i(t_k) + sum alpha_{j,i}(t_k) exp(-gamma_{j,i}(t_k) delta_t)`.
- Mini-batch compensator integration for the exponential Hawkes kernel.
- Streaming compensator integration across batch boundaries without storing all historical events.
- Chronological sorting inside the loss to prevent order-dependent likelihood errors.
- Explicit Hawkes state resets at train, validation, test, and shuffled-time sanity-check boundaries.
- Positive constraints for `mu`, `alpha`, and `gamma`.
- Streaming Hawkes NLL wired into `train.py`, `evaluate.py`, `dry_run.py`, and `run_diagnostics.py`.
- Focused tests for differentiability, chronological invariance, split-stream equivalence, reset behavior, and positive parameter emission.
- A trainable static multivariate Hawkes baseline with global `mu`, GDELT response weights, cross-platform `alpha`, and decay `gamma`.
- A baseline runner with chronological train/validation/test splits, gradient clipping, validation NLL reporting, and optional JSON export.
- A Hawkes residual cascade monitor that scores each event with intensity, compensator, event NLL, excitation mass, rolling z-score, and alert level.
- A monitor CLI that exports top anomalous cascade events and optional per-event CSV telemetry.

This moves the prototype closer to the whitepaper goal: modeling not only whether a narrative becomes viral, but how activity on one platform excites future activity on another.

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch Geometric can require a wheel matched to your PyTorch and CUDA setup. If the generic install fails, use the official selector at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html.

## Usage

Generate synthetic data:

```bash
python data/synthetic_generator.py --events 10000 --out data/synthetic_events.pkl
```

Run a model smoke test:

```bash
python dry_run.py
```

Train the model:

```bash
python train.py
```

Train the static Hawkes baseline:

```bash
python run_hawkes_baseline.py --data data/synthetic_events.pkl --epochs 8 --out baselines/hawkes_report.json
```

Run cascade anomaly monitoring:

```bash
python run_cascade_monitor.py --data data/synthetic_events.pkl --epochs 4 --out baselines/cascade_alerts.json --events-out baselines/cascade_events.csv
```

Run tests:

```bash
python -m pytest tests -q
```

## Verification

Completed successfully:

```bash
python -m pytest tests/test_hawkes_loss.py -q
python -m pytest tests/test_static_hawkes_baseline.py -q
python -m pytest tests/test_cascade_monitor.py -q
python -m py_compile analysis_tools/cascade_monitor.py run_cascade_monitor.py run_hawkes_baseline.py models/hawkes.py models/virality_head.py models/loss.py evaluate.py train.py dry_run.py run_diagnostics.py data/dataset.py
```

Current environment caveat: `python dry_run.py` could not be executed here because `torch_geometric` is not installed in this workspace Python environment.

The committed `data/synthetic_events.pkl` may also need regeneration if your NumPy version cannot unpickle it. Use:

```bash
python data/synthetic_generator.py --events 10000 --out data/synthetic_events.pkl
```

## Research Notes

The new Hawkes loss, static baseline, and residual monitor are practical cascade-modeling benchmarks. A full production research pass should next add:

- Calibration metrics for virality risk thresholds.
- Real ingestion from Hacker News, Reddit, RSS, and GDELT as outlined in [internet twin.md](./internet%20twin.md).
