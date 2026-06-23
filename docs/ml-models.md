# ML Models

## Model Architecture

### Temporal Graph Network (TGN)
- **File**: `models/tgn_core.py`
- **Purpose**: Learns dynamic node representations from event streams
- **Architecture**: Memory module + message aggregator + temporal attention
- **Input**: Graph snapshots with temporal edges
- **Output**: Node embeddings (128-dim) used by downstream heads

### Virality Head
- **File**: `models/virality_head.py`
- **Purpose**: Multi-task prediction from TGN embeddings
- **Outputs**: SEIR parameters (β, ζ), Hawkes parameters (λ₀, α, β_h), engagement score
- **Architecture**: MLP with 3 output heads

### Neural Hawkes Process
- **File**: `models/hawkes.py`
- **Purpose**: Models self-exciting event cascades across platforms
- **Architecture**: Cross-platform excitation matrix + exponential decay
- **Key equation**: λ(t) = μ + Σ α·exp(-β(t-tᵢ))

### HMF Bridge
- **File**: `models/hmf_bridge.py`
- **Purpose**: Bridges microscopic TGN predictions to macroscopic SEIR-Z-D parameters
- **Architecture**: Aggregates node-level predictions → population-level rates

### Deffuant Opinion Model
- **File**: `models/deffuant.py`
- **Purpose**: Bounded-confidence opinion dynamics
- **Variant**: SmoothDeffuant with tanh-bounded convergence and attention scaling

### Influence Module
- **File**: `models/influence.py`
- **Purpose**: Dynamic influence scoring I(v,t) = ω₁·TPR + ω₂·dE/dt + ω₃·C_D

### Multi-Task Loss
- **File**: `models/loss.py`
- **Purpose**: Homoscedastic uncertainty weighting across all loss components
- **Components**: SEIR loss, Hawkes NLL, engagement MSE, influence loss

## Forecasting Models

### Virality Forecaster
- **File**: `ml/forecasting/virality_forecaster.py`
- **Purpose**: Multi-horizon engagement prediction (6h/24h/72h)
- **Method**: Branching process + exponential decay + velocity correction

### Trend Predictor
- **File**: `ml/forecasting/trend_predictor.py`
- **Purpose**: Trend rank prediction with momentum and breakout detection

### Polarization Model
- **File**: `ml/forecasting/polarization_model.py`
- **Purpose**: Quantifies opinion polarization using Esteban-Ray index, bimodality, echo chamber metrics

## Baselines
- **Static MHP**: `baselines/mhp.py` — Multivariate Hawkes Process
- **Static GNN**: `baselines/static_gnn.py` — Fixed-topology GCN baseline

## MLOps (Training, Tracking & Retraining)

The `ml/training/` and `ml/registry/` packages provide a production training
lifecycle. All components degrade gracefully to a local JSON store when MLflow
is not installed, so they run in CI and offline.

### Experiment Tracking & Pipeline
- **File**: `ml/training/pipeline.py`
- **`ExperimentTracker`**: thin façade over MLflow (params/metrics/artifacts) with a local fallback.
- **`TrainingPipeline`**: runs a tracked training pass, evaluates it, and **auto-promotes** the new version to `Production` only if it beats the incumbent on the primary metric.

### Model Registry
- **File**: `ml/registry/model_registry.py`
- Register versions, transition stages (`Staging` → `Production` with auto-archive), and resolve the current production model. MLflow-backed or local JSON.

### Drift Detection & Scheduled Retraining
- **File**: `ml/training/scheduler.py`
- **`RetrainingScheduler`**: triggers a retrain when the production model degrades beyond a relative threshold **or** exceeds a max staleness window.
- Wired to **`.github/workflows/retrain.yml`** for nightly runs.

### Evaluation
- **File**: `ml/training/evaluate.py`
- NumPy-only `regression_metrics` / `classification_metrics` and `ModelEvaluator` producing comparable `EvaluationReport`s.

### CLI
```bash
# Run a tracked training pass and auto-promote if better (no PyG needed):
python run_training_pipeline.py train --demo

# Nightly drift check + conditional retrain:
python run_training_pipeline.py schedule --observed-mae 0.55 --demo
```

See `docs/mlops.md` for the full lifecycle walkthrough.
