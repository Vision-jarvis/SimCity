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

## Monitoring

### Prometheus
- URL: `http://localhost:9090`
- Config: `monitoring/prometheus.yml`

### Grafana
- URL: `http://localhost:3001`
- Dashboards: `monitoring/grafana/dashboards/`
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
