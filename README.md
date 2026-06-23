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

## MLOps & Deployment

The roadmap's Phase 3 (MLOps) and Phase 6 (production hardening) scaffolding are now in place:

- `ml/registry/model_registry.py` — MLflow-backed model registry with a zero-dependency local JSON fallback (register, stage promotion, production lookup).
- `ml/training/pipeline.py` — `ExperimentTracker` (MLflow or local) + `TrainingPipeline` that logs params/metrics/artifacts and auto-promotes a run when it beats the current Production model.
- `ml/training/scheduler.py` — `RetrainingScheduler` with drift detection (performance-degradation + staleness gates).
- `run_training_pipeline.py` — CLI: `train` and `schedule` subcommands (use `--demo` to exercise the full track→register→promote flow without PyTorch Geometric).
- `.github/workflows/retrain.yml` — nightly drift-check + conditional retrain.

Run the MLOps demo (no GPU / PyG required):

```bash
python run_training_pipeline.py train --demo
python run_training_pipeline.py schedule --observed-mae 0.55 --demo
```

Deployment artifacts:

- `Dockerfile` (API, CPU-only) and `frontend/Dockerfile` (multi-stage Next.js).
- `infra/terraform/` — Oracle Cloud Always-Free VM + k3s bootstrap (4 OCPU / 24 GB, $0).
- `infra/k8s/` — kustomize manifests (Kafka, Neo4j, Redis, API, frontend); apply with `kubectl apply -k infra/k8s`.
- `infra/helm/values.yaml` — Helm configuration surface.
- `scripts/locustfile.py` — Locust load test (`locust -f scripts/locustfile.py --host http://localhost:8000`).
- `monitoring/grafana/dashboards/model_performance.json` — model latency/MAE/drift dashboard.

See [infra/README.md](./infra/README.md) for the full deploy walkthrough.

## Counterfactual Intervention Simulator

`simulation/intervention.py` turns the forward SEIR-Z-D simulator into a digital-twin "what-if" engine: it runs a deterministic **baseline** and a **counterfactual** that differ only by an applied intervention, then reports the deltas (peak misinformation, total reach, persistent zealots, debunked volume, polarization).

Intervention types: `fact_check`, `counter_narrative`, `deplatform_bots`, `rate_limit`, `influencer_amplify`. Exposed via `POST /simulate/intervention` and the **Intervention** page in the frontend (baseline-vs-treatment delta cards + overlay chart). See [docs/simulation-guide.md](docs/simulation-guide.md).

## Cross-Platform Narrative Transfer

`analysis_tools/narrative_tracker.py` clusters multi-platform events into narratives and reconstructs how each one propagates across platforms (e.g. Reddit → Hacker News → News/GDELT) with per-hop time lags and a content **mutation score**. Similarity is pluggable (dependency-free token-Jaccard by default; inject embeddings for semantic clustering). Exposed via `POST /trends/narrative-transfer`.

## Data Sources

Seven ingesters under `ingestion/sources/`: Reddit, Hacker News, GDELT, RSS, YouTube, **Wikipedia** (Wikimedia recent-changes — live edit activity), and **Bluesky** (AT Protocol public AppView — reliable, key-optional; the dependable free X-alternative). All are key-optional or free-tier. See [docs/data-sources.md](docs/data-sources.md).

## Demo Notebooks

Colab-ready notebooks in `notebooks/` cover data exploration, the TGN prototype, virality forecasting, the multi-agent simulation, and narrative tracking.

## Research Notes

The Hawkes loss, static baseline, and residual monitor are practical cascade-modeling benchmarks. A full production research pass should next add:

- Calibration metrics for virality risk thresholds.
- Real ingestion from Hacker News, Reddit, RSS, and GDELT as outlined in [internet twin.md](./internet%20twin.md).
- Wiring `train.py` to return final metrics so the MLflow pipeline logs real (not placeholder) numbers.
