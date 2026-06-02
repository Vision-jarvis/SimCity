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
