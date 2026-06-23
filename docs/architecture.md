# Architecture Overview

## System Architecture

SimCity is a hybrid-model engine for simulating internet behavior, combining:

1. **Real-time Data Ingestion** — Multi-platform event streams (Reddit, HN, GDELT, RSS, YouTube, Wikipedia, Bluesky)
2. **Knowledge Graph** — Neo4j-backed graph tracking entities, relationships, and influence
3. **NLP Intelligence** — Transformer-based text analysis pipeline
4. **GNN/ML Core** — Temporal Graph Networks + Neural Hawkes for virality prediction
5. **Simulation Engine** — SEIR-Z-D epidemiological model coupled with opinion dynamics
6. **API Layer** — FastAPI REST + GraphQL + WebSocket
7. **Frontend Dashboard** — Next.js with 3D graph visualization and real-time streaming
8. **MLOps** — MLflow experiment tracking, model registry, drift-gated retraining
9. **Infrastructure** — Docker images, Terraform (OCI Always-Free) + k3s manifests, Prometheus/Grafana

```
┌──────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                       │
│ Dashboard│Graph3D│Simulate│Intervention│Trends│Narratives     │
└─────────────────────────┬────────────────────────────────────┘
                          │ REST / GraphQL / WebSocket
┌─────────────────────────┴────────────────────────────────────┐
│                      API Layer (FastAPI)                       │
│  /graph  /simulate  /trends  /search  /ws  /graphql           │
└───┬────────┬──────────┬─────────┬──────────┬─────────────────┘
    │        │          │         │          │
┌───┴────┐ ┌─┴──────┐ ┌┴───────┐ │   ┌──────┴──────┐
│ Neo4j  │ │ Engine │ │  ML    │ │   │  NLP        │
│ Graph  │ │ SEIR-ZD│ │ TGN    │ │   │ Sentiment   │
│ Builder│ │ Agents │ │ Hawkes │ │   │ Toxicity    │
└───┬────┘ └────────┘ │ Deffuan│ │   │ Embeddings  │
    │                 └────────┘ │   └─────────────┘
┌───┴────────────────────────────┴─────────────────────────────┐
│                    Kafka Event Bus                             │
└───┬──────┬──────┬──────┬──────┬──────────────────────────────┘
    │     │     │     │     │     │     │
┌───┴─┐┌──┴─┐┌──┴──┐┌─┴─┐┌──┴─┐┌──┴─┐┌─┴─────┐
│Reddit││ HN ││GDELT││RSS││ YT ││Wiki││Bluesky│
└──────┘└────┘└─────┘└───┘└────┘└────┘└───────┘
```

## Data Flow

1. **Ingesters** poll platform APIs and produce events to Kafka
2. **Processors** (dedup → normalize → enrich) clean and annotate events
3. **Graph Builder** consumes from Kafka and creates Neo4j nodes + edges
4. **Simulation Engine** runs SEIR-Z-D forward models on demand
5. **ML Models** predict virality, engagement, and polarization
6. **API** exposes all capabilities via REST + GraphQL + WebSocket
7. **Frontend** visualizes in real-time

## MLOps & Deployment Flow

```
train.py / TrainingPipeline ──► ExperimentTracker (MLflow or local)
        │                                │
        ▼                                ▼
  EvaluationReport ──► ModelRegistry (Staging → Production)
        │                                │
        ▼                                ▼
  RetrainingScheduler ◄── drift / staleness checks (nightly CI)

Docker images ──► CI (deploy.yml) ──► Terraform (OCI VM + k3s)
                                          │
                                          ▼
                          kubectl apply -k infra/k8s
                          (Kafka, Neo4j, Redis, API, frontend)
                                          │
                                          ▼
                          Prometheus + Grafana dashboards
```

## Key Design Decisions

- **Lazy-loading ML models** — NLP models loaded only when first accessed, enabling fast startup
- **Graceful degradation** — System works without Neo4j/Kafka/ML models running
- **Mock data fallback** — All API endpoints return structured mock data when backends are offline
- **Free-tier only** — All components are open-source or free-tier compatible
