# Deployment Guide

## Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+

## Development Setup

```bash
# 1. Clone and configure
git clone <repo>
cd SimCity
cp .env.example .env
# Edit .env with your API keys

# 2. Start infrastructure
docker-compose up -d

# 3. Install Python dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 4. Seed the graph
python scripts/seed_graph.py

# 5. Start API
uvicorn api.main:app --reload --port 8000

# 6. Start Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## Production Deployment

### Docker Compose
```bash
docker-compose -f docker-compose.prod.yml up -d
```

This starts:
- Kafka + Zookeeper (messaging)
- Neo4j (graph database)
- Redis (caching)
- API service (FastAPI)
- Frontend (Next.js)
- Grafana + Prometheus (monitoring)

### Environment Variables
See `.env.example` for all required variables.

### Container Images
The stack builds from two Dockerfiles:
- `Dockerfile` — API (CPU-only PyTorch, sized for free-tier inference).
- `frontend/Dockerfile` — multi-stage Next.js production build.

```bash
docker build -t simcity-api:latest .
docker build -t simcity-frontend:latest ./frontend
```

## Zero-Cost Cloud Deployment (Oracle Always-Free + k3s)

Full Infrastructure-as-Code lives under `infra/` (see `infra/README.md`):

```bash
# 1. Provision a 4 OCPU / 24 GB Always-Free VM + network + k3s bootstrap
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in OCI creds
terraform init && terraform apply
terraform output service_urls

# 2. Deploy the stack to the k3s node
kubectl apply -k infra/k8s
```

- `infra/terraform/` — OCI VM, VCN, security list, cloud-init k3s install.
- `infra/k8s/` — kustomize manifests: Kafka, Neo4j (PVC), Redis, API (probes), frontend (NodePort).
- `infra/helm/values.yaml` — single tunable surface for a Helm-based install.

Estimated monthly cost: **$0** (entirely within OCI Always-Free allowances).

## Load Testing

```bash
pip install locust
uvicorn api.main:app                      # in one shell
locust -f scripts/locustfile.py --host http://localhost:8000

# Headless / CI benchmark:
locust -f scripts/locustfile.py --host http://localhost:8000 \
    --headless -u 50 -r 10 -t 1m --csv .locust_results/run
```

Validate against the scalability targets in `internet twin.md`
(e.g. inference latency < 2s → < 500ms; throughput 1K → 50K events/min).

## Monitoring

### Prometheus
- URL: `http://localhost:9090`
- Config: `monitoring/prometheus.yml`

### Grafana
- URL: `http://localhost:3001`
- Dashboards: `monitoring/grafana/dashboards/`
  - `simcity_overview.json` — API rate, simulations, Kafka lag, Neo4j health.
  - `model_performance.json` — virality MAE, Hawkes NLL, inference latency p95, drift, model staleness.
- Default login: admin/admin

### Alerts
- Config: `monitoring/alerts/rules.yml`
- Monitors: API errors, Kafka lag, Neo4j health, memory usage

## Makefile Commands

```bash
make install    # Full setup
make dev        # Start everything
make test       # Run all tests
make lint       # Run linters
make smoke      # Quick validation
make benchmark  # Performance tests
make clean      # Clean caches
```
