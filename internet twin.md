# 🌐 AI Digital Twin of the Internet

> A real-time autonomous simulation engine for internet behavior, virality, influence cascades, and collective dynamics — built entirely on free and open-source infrastructure.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Implementation Status](#-implementation-status)
- [Zero-Cost Strategy](#-zero-cost-strategy)
- [Architecture](#-architecture)
- [File Structure](#-file-structure)
- [Tech Stack (Free Only)](#-tech-stack-free-only)
- [Roadmap](#-roadmap)
- [Phase-by-Phase Requirements](#-phase-by-phase-requirements)
- [Setup & Installation](#-setup--installation)
- [Data Sources](#-data-sources-free-tier--open)
- [ML Models](#-ml-models-open-source-only)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Project Overview

This platform is a **live computational mirror of the internet** — a distributed AI system that ingests real-time internet data, builds a dynamic knowledge graph, runs multi-agent simulations, and forecasts viral trends, misinformation outbreaks, and collective sentiment shifts.

Think: **SimCity for the Internet, powered by AI.**

### Core Capabilities

| Capability | Description |
|---|---|
| Real-Time Graph | Live knowledge graph of users, topics, communities, influencers |
| Virality Forecasting | Predict meme explosions before mainstream adoption |
| Narrative Tracking | Track how ideas mutate across platforms |
| Intervention Simulator | "What if influencer X posts about Y?" |
| Multi-Agent Simulation | Autonomous agents mimicking real internet actors |
| Misinformation Detection | Identify and track false narrative propagation |
| Geographic Sentiment | Regional ideology and emotion shift mapping |

---

## ✅ Implementation Status

This document is the original vision. The repository now implements the large
majority of it. Highlights and divergences from the plan below:

**Built and working** (see `README.md` and `docs/`):
- 7 ingesters: Reddit, Hacker News, GDELT, RSS, YouTube, **Wikipedia**, **Bluesky** (AT Protocol).
- NLP suite: embeddings, topic, sentiment, toxicity, stance, misinformation, summarizer.
- Graph engine: Neo4j client (degrades gracefully when offline), node/edge types, Louvain, PageRank/betweenness.
- Modeling core under `models/`: Temporal Graph Network, neural Hawkes cross-platform excitation, virality head, Deffuant opinion dynamics, HMF bridge — plus `ml/forecasting/` and Hawkes baselines.
- **MLOps** (`ml/training/`, `ml/registry/`): MLflow tracking + model registry + drift-gated retraining (local-JSON fallback when MLflow absent), nightly `retrain.yml`.
- Simulation: SEIR-Z-D engine, LangGraph multi-agent runtime (Influencer/Bot/Skeptic/Community/News), scenario presets, and a **counterfactual intervention simulator** (`simulation/intervention.py`).
- **Cross-platform narrative-transfer detection** (`analysis_tools/narrative_tracker.py`).
- FastAPI (REST + GraphQL + WebSocket), Next.js frontend (Dashboard, Graph3D, Simulate, Intervention, Trends, Narratives).
- Infra: Dockerfiles, Terraform (OCI Always-Free) + k3s manifests + Helm, Prometheus/Grafana, Locust load test, 5 Colab notebooks.

**Divergences from the plan:**
- The GNN lives in `models/` (repo root), not `ml/gnn/`.
- Agent personas are consolidated in `simulation/agents/personas.py` rather than one file per agent.
- Twitter/X is ingested via **Bluesky** (reliable, free) instead of Nitter (most mirrors are dead).
- Forecasting uses an analytical Hawkes-SEIR model, not a Temporal Fusion Transformer (TFT).
- `requirements-test.txt` is the lean CI dependency set; the full `requirements.txt` is for production.

**Not yet implemented:** PostgreSQL enrichment store, persisted temporal graph snapshots, TFT forecaster, security audit (auth/rate-limiting).

---

## 💰 Zero-Cost Strategy

Every component in this project uses **free, open-source, or free-tier** services only. No paid APIs. No paid cloud infra. Here is how:

### Infrastructure (Free)

| Need | Free Solution |
|---|---|
| Cloud compute | Google Colab (free GPU), Kaggle Notebooks (free GPU), Oracle Cloud Free Tier (4 OCPU, 24GB RAM always-free), Hugging Face Spaces |
| Object storage | Hugging Face Datasets repo, Cloudflare R2 (10GB free), Google Drive (15GB) |
| Kubernetes | k3s locally, or Civo free $250 credit, or Fly.io free tier |
| CI/CD | GitHub Actions (2000 min/month free) |
| Monitoring | Grafana Cloud free tier, Prometheus self-hosted |
| Database | Supabase free tier (PostgreSQL), self-hosted Neo4j Community Edition |
| Vector DB | Qdrant self-hosted (Docker), ChromaDB (in-process, free) |

### Data Sources (Free)

| Source | Free Access Method |
|---|---|
| Reddit | Official Reddit API (free, 100 req/min) |
| Hacker News | Algolia HN Search API (completely free, unlimited) |
| News | GDELT Project (completely free, real-time) |
| RSS feeds | Any public RSS (free) |
| Wikipedia trends | Wikimedia API (free) |
| YouTube | YouTube Data API v3 (10,000 free units/day) |
| Twitter/X | **No free API** — use **Bluesky** (AT Protocol public AppView, free, no key) as the social signal |

### ML Models (Free)

| Model Type | Free Option |
|---|---|
| Text embeddings | `sentence-transformers` (all-MiniLM-L6-v2, BGE-small) — run locally |
| LLM inference | Ollama (local), Groq free tier (6000 tokens/min), Together AI free credits |
| NLP tasks | `transformers` (HuggingFace) — run locally or on Colab |
| Graph ML | PyG (PyTorch Geometric) — fully open source |

### Estimated Monthly Cost: **$0**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Data Sources (Free)                        │
│   Reddit API  │  GDELT  │  HN Algolia  │  RSS Feeds  │  YouTube  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Ingestion Layer                                 │
│       Kafka (self-hosted)  +  Scrapy  +  Async Python workers    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  NLP & Feature Engineering                        │
│  HuggingFace Transformers  │  sentence-transformers  │  spaCy    │
└──────────┬───────────────────────────────────┬───────────────────┘
           │                                   │
           ▼                                   ▼
┌─────────────────────┐             ┌─────────────────────┐
│   Graph Engine      │             │   Forecasting AI    │
│  Neo4j Community    │             │   PyTorch + PyG     │
│  NetworkX           │             │   TFT / Prophet     │
└─────────┬───────────┘             └──────────┬──────────┘
          │                                    │
          └───────────────┬────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│               Multi-Agent Simulation Engine                       │
│                  LangGraph  +  Ray (local)                        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    API Layer                                       │
│              FastAPI  +  WebSockets  +  GraphQL (Strawberry)     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                 Frontend Visualization                            │
│          Next.js  │  Three.js  │  D3.js  │  Zustand              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

> **Note:** the tree below is the original proposed layout. The actual
> repository follows it closely with a few divergences (see
> [Implementation Status](#-implementation-status)): the GNN lives in `models/`
> rather than `ml/gnn/`; agents are consolidated in
> `simulation/agents/personas.py`; and the repo adds `ml/training/`,
> `ml/registry/`, `infra/`, `notebooks/`, `analysis_tools/narrative_tracker.py`,
> `simulation/intervention.py`, Wikipedia/Bluesky ingesters, and
> `requirements-test.txt`.

```
ai-digital-twin/
│
├── README.md
├── .env.example                    # Environment variable template (no secrets committed)
├── .gitignore
├── docker-compose.yml              # Local dev: Kafka, Neo4j, Redis, API
├── docker-compose.prod.yml         # Production stack
├── Makefile                        # Convenience commands
│
├── docs/
│   ├── architecture.md
│   ├── data-sources.md
│   ├── ml-models.md
│   ├── api-reference.md
│   ├── simulation-guide.md
│   └── deployment.md
│
├── infra/
│   ├── terraform/                  # IaC (Oracle Cloud Free Tier)
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── k8s/                        # k3s / Kubernetes manifests
│   │   ├── kafka/
│   │   ├── neo4j/
│   │   ├── api/
│   │   └── frontend/
│   └── helm/
│       └── values.yaml
│
├── ingestion/                      # Data ingestion layer
│   ├── __init__.py
│   ├── config.py
│   ├── kafka_producer.py
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── reddit_ingester.py
│   │   ├── hackernews_ingester.py
│   │   ├── gdelt_ingester.py
│   │   ├── rss_ingester.py
│   │   └── youtube_ingester.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── deduplicator.py
│   │   ├── normalizer.py
│   │   └── enricher.py
│   └── tests/
│       └── test_ingesters.py
│
├── nlp/                            # NLP & semantic intelligence
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py               # sentence-transformers wrapper
│   ├── topic_extractor.py          # BERTopic
│   ├── sentiment_analyzer.py       # cardiffnlp/twitter-roberta
│   ├── toxicity_classifier.py
│   ├── stance_detector.py
│   ├── misinformation_scorer.py
│   ├── summarizer.py               # transformers pipeline
│   └── tests/
│       └── test_nlp.py
│
├── graph/                          # Dynamic internet graph engine
│   ├── __init__.py
│   ├── config.py
│   ├── neo4j_client.py
│   ├── graph_builder.py            # Builds/updates the knowledge graph
│   ├── node_types.py               # User, Topic, Community, etc.
│   ├── edge_types.py               # Influence, Repost, Narrative, etc.
│   ├── influence_scorer.py         # PageRank, betweenness
│   ├── community_detector.py       # Louvain, Label Propagation
│   └── tests/
│       └── test_graph.py
│
├── ml/                             # ML models
│   ├── __init__.py
│   ├── config.py
│   ├── gnn/
│   │   ├── __init__.py
│   │   ├── temporal_gnn.py         # PyG temporal graph model
│   │   ├── graph_attention.py      # GAT for influence
│   │   ├── diffusion_model.py      # Info propagation
│   │   └── trainer.py
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── virality_forecaster.py  # TFT / Prophet
│   │   ├── trend_predictor.py
│   │   ├── polarization_model.py
│   │   └── trainer.py
│   ├── training/
│   │   ├── pipeline.py
│   │   ├── scheduler.py            # MLflow-tracked scheduled retraining
│   │   └── evaluate.py
│   └── registry/
│       └── model_registry.py       # MLflow model registry
│
├── simulation/                     # Multi-agent simulation engine
│   ├── __init__.py
│   ├── config.py
│   ├── engine.py                   # Main simulation runner
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── influencer_agent.py
│   │   ├── community_agent.py
│   │   ├── bot_agent.py
│   │   ├── news_agent.py
│   │   └── skeptic_agent.py
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── scenario_builder.py
│   │   └── presets/
│   │       ├── influencer_tweet.py
│   │       ├── misinfo_outbreak.py
│   │       └── platform_outage.py
│   └── tests/
│       └── test_simulation.py
│
├── api/                            # FastAPI backend
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── graph.py                # Graph query endpoints
│   │   ├── simulation.py           # Scenario run endpoints
│   │   ├── trends.py               # Forecasting endpoints
│   │   ├── search.py               # Semantic search
│   │   └── websocket.py            # Real-time stream
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── graph_schemas.py
│   │   ├── simulation_schemas.py
│   │   └── trend_schemas.py
│   ├── graphql/
│   │   ├── __init__.py
│   │   └── schema.py               # Strawberry GraphQL schema
│   └── tests/
│       └── test_api.py
│
├── frontend/                       # Next.js visualization
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── public/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx            # Dashboard home
│   │   │   ├── graph/page.tsx      # 3D live graph
│   │   │   ├── simulate/page.tsx   # Scenario simulator
│   │   │   ├── trends/page.tsx     # Virality forecasts
│   │   │   └── narratives/page.tsx # Narrative tracker
│   │   ├── components/
│   │   │   ├── Graph3D.tsx         # Three.js graph
│   │   │   ├── VitalityHeatmap.tsx # D3 heatmap
│   │   │   ├── NarrativeTimeline.tsx
│   │   │   ├── GeoSentimentMap.tsx
│   │   │   └── SimulationPlayer.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   └── useGraphData.ts
│   │   └── store/
│   │       └── useStore.ts         # Zustand store
│   └── tests/
│
├── scripts/
│   ├── seed_graph.py               # Populate initial graph data
│   ├── backfill_gdelt.py           # Historical GDELT backfill
│   ├── benchmark.py                # System performance benchmarks
│   └── smoke_test.py               # End-to-end smoke test
│
├── notebooks/                      # Jupyter notebooks (Colab-ready)
│   ├── 01_data_exploration.ipynb
│   ├── 02_gnn_prototype.ipynb
│   ├── 03_virality_model.ipynb
│   ├── 04_simulation_demo.ipynb
│   └── 05_narrative_tracking.ipynb
│
├── .github/
│   └── workflows/
│       ├── test.yml                # Run tests on PR
│       ├── lint.yml                # Ruff + mypy
│       └── deploy.yml              # Deploy on merge to main
│
└── monitoring/
    ├── prometheus.yml
    ├── grafana/
    │   └── dashboards/
    │       ├── ingestion.json
    │       ├── model_performance.json
    │       └── api_health.json
    └── alerts/
        └── rules.yml
```

---

## 🛠️ Tech Stack (Free Only)

### Backend & Streaming

| Component | Technology | Cost |
|---|---|---|
| API framework | FastAPI | Free |
| GraphQL | Strawberry | Free |
| Message bus | Apache Kafka (self-hosted Docker) | Free |
| Stream processing | Python Kafka consumers | Free |
| Task queue | Celery + Redis | Free |
| Web scraping | Scrapy + httpx | Free |

### Databases

| Component | Technology | Cost |
|---|---|---|
| Primary DB | PostgreSQL via Supabase free tier | Free |
| Graph DB | Neo4j Community Edition (self-hosted) | Free |
| Vector DB | ChromaDB (in-process) or Qdrant (Docker) | Free |
| Cache | Redis (Docker) | Free |
| Object store | Cloudflare R2 (10GB) or local disk | Free |

### ML & AI

| Component | Technology | Cost |
|---|---|---|
| Deep learning | PyTorch | Free |
| Graph ML | PyTorch Geometric (PyG) | Free |
| NLP | HuggingFace Transformers | Free |
| Embeddings | sentence-transformers | Free |
| Topic modeling | BERTopic | Free |
| Time-series | NeuralForecast / Prophet | Free |
| LLM inference | Ollama (local) or Groq free tier | Free |
| Experiment tracking | MLflow (self-hosted) | Free |
| Agent framework | LangGraph | Free |
| Distributed compute | Ray (local mode) | Free |

### Frontend

| Component | Technology | Cost |
|---|---|---|
| Framework | Next.js | Free |
| 3D graph | Three.js | Free |
| Charts | D3.js | Free |
| Styling | Tailwind CSS | Free |
| State | Zustand | Free |
| Real-time | Native WebSockets | Free |

### DevOps

| Component | Technology | Cost |
|---|---|---|
| Containers | Docker + Docker Compose | Free |
| Orchestration | k3s (lightweight Kubernetes) | Free |
| CI/CD | GitHub Actions | Free (2000 min/mo) |
| IaC | Terraform | Free |
| Monitoring | Prometheus + Grafana (self-hosted) | Free |
| Hosting option A | Oracle Cloud Always-Free (4 OCPU, 24GB) | Free |
| Hosting option B | Fly.io free tier | Free |
| Hosting option C | Hugging Face Spaces (Gradio/Docker) | Free |

---

## 🗺️ Roadmap

### Phase 0 — Foundation (Weeks 1–2)
> Goal: Repo, dev environment, data flowing, basic graph up.

- [x] Initialize repository, set up `.env.example`, `docker-compose.yml`
- [x] Implement Reddit and Hacker News ingesters
- [x] Set up Kafka with Docker Compose
- [x] Implement basic NLP pipeline (embeddings + sentiment)
- [x] Spin up Neo4j Community Edition
- [x] Build basic graph schema (Users, Topics, Communities)
- [x] Write basic FastAPI endpoints (`/health`, `/graph/nodes`, `/graph/edges`)
- [x] Seed graph script (`scripts/seed_graph.py`, `scripts/backfill_gdelt.py`)
- [x] Set up GitHub Actions CI (lint + test)

**Milestone:** Live data flowing into graph, queryable via API.

---

### Phase 1 — NLP Intelligence (Weeks 3–4)
> Goal: Every incoming post enriched with NLP metadata.

- [x] BERTopic topic modeling on incoming posts
- [x] Sentiment analysis pipeline (cardiffnlp/twitter-roberta)
- [x] Toxicity classifier (Detoxify)
- [x] Stance detection model
- [x] Misinformation probability scorer (heuristic scorer in `nlp/`)
- [x] Semantic clustering with sentence-transformers
- [ ] Store enriched records in PostgreSQL + vector index in ChromaDB
- [x] Add `/nlp/analyze` API endpoint (`/search/nlp/analyze`)

**Milestone:** Every post tagged with topic, sentiment, toxicity, virality signal.

---

### Phase 2 — Graph Intelligence (Weeks 5–7)
> Goal: Live, evolving internet knowledge graph.

- [x] Build graph update pipeline (Kafka consumer → Neo4j writer)
- [x] Implement node types: User, Topic, Community, Hashtag, Organization
- [x] Implement edge types: Influence, Repost, Reply, Narrative Transfer
- [x] Community detection with Louvain algorithm
- [x] Influence scoring with PageRank + betweenness centrality
- [ ] Build temporal graph snapshots (graph at time T)
- [x] Add GDELT news ingester
- [x] Cross-platform narrative transfer detection (`analysis_tools/narrative_tracker.py`, `/trends/narrative-transfer`)
- [x] Graph query API endpoints (Cypher-backed)

**Milestone:** A living, queryable graph of ~100K+ nodes and edges.

---

### Phase 3 — GNN & Forecasting (Weeks 8–11)
> Goal: Predict virality and cascade propagation.

- [x] Implement Temporal GNN for influence spread (PyG) — `models/tgn_core.py`
- [x] Train virality classifier (`models/virality_head.py`, `train.py`)
- [x] Graph attention for high-impact nodes (TransformerConv in TGN + influence scorer)
- [x] Build diffusion model for idea propagation (Hawkes + SEIR-Z-D + Deffuant)
- [ ] Implement Temporal Fusion Transformer (forecaster is analytical Hawkes-SEIR, not TFT)
- [x] Build trend prediction pipeline (next 6h/24h/72h) — `ml/forecasting/`
- [x] Set up MLflow for experiment tracking (`ml/training/`, `ml/registry/`)
- [x] Build scheduled retraining pipeline (GitHub Actions nightly) — `retrain.yml`
- [x] Model evaluation dashboard (Grafana) — `monitoring/grafana/dashboards/model_performance.json`

**Milestone:** Virality score + predicted reach for any incoming topic.

---

### Phase 4 — Simulation Engine (Weeks 12–15)
> Goal: Run "what-if" internet scenarios.

- [x] Implement base agent class with behavioral parameters
- [x] Build Influencer, Community, Bot, News, Skeptic agents
- [x] Implement LangGraph-based agent runtime
- [x] Build scenario configuration system (`simulation/scenario_builder.py`)
- [x] Preset scenarios: influencer tweet, misinfo outbreak, platform outage
- [x] Simulation output: propagation map, virality curve, polarization delta
- [x] Simulation result caching and replay
- [x] Simulation API endpoints (`/simulate/run`, `/simulate/replay`)
- [x] **Counterfactual intervention simulator** (`/simulate/intervention`) — baseline vs. what-if

**Milestone:** Fully runnable "what-if" scenarios with multi-agent dynamics.

---

### Phase 5 — Frontend Visualization (Weeks 16–19)
> Goal: Make it visually stunning and interactive.

- [x] Set up Next.js project with Tailwind + Zustand
- [x] Real-time 3D internet graph (Three.js + WebSocket)
- [x] Virality heatmap (D3.js)
- [x] Narrative evolution timeline
- [x] Geographic sentiment map (D3 + GeoJSON)
- [x] Cascade simulation playback (animation)
- [x] Scenario simulator UI (form → run → watch) + intervention comparison page
- [x] Live trending topics feed

**Milestone:** Fully interactive visualization dashboard.

---

### Phase 6 — Production Hardening (Weeks 20–22)
> Goal: Reliable, monitored, deployable system.

- [x] Deploy to Oracle Cloud Free Tier (Terraform + k3s) — `infra/`
- [x] Set up Prometheus metrics across all services
- [x] Build Grafana dashboards (overview + model performance)
- [x] Configure alerting rules (`monitoring/alerts/rules.yml`)
- [x] Load testing (Locust) — `scripts/locustfile.py`
- [ ] Security audit (auth, rate limiting, secrets management)
- [x] Full README and API documentation (`docs/`)
- [x] Create 5 demo notebooks (Colab-ready) — `notebooks/`

**Milestone:** Production-grade, monitored, zero-cost deployment.

---

## 📋 Phase-by-Phase Requirements

### Phase 0 Requirements

**Python packages:**
```
kafka-python
praw                    # Reddit API
requests
httpx
neo4j
fastapi
uvicorn
pydantic
python-dotenv
```

**Services (Docker):**
- Apache Kafka + Zookeeper
- Neo4j Community Edition
- Redis

---

### Phase 1 Requirements

**Python packages:**
```
transformers
sentence-transformers
bertopic
detoxify
torch
spacy
```

**Free API keys needed:**
- Reddit API (free at reddit.com/prefs/apps)
- YouTube Data API v3 (free at console.cloud.google.com)

---

### Phase 2 Requirements

**Python packages:**
```
python-louvain
networkx
neo4j
gdelt               # GDELT wrapper
feedparser          # RSS feeds
```

---

### Phase 3 Requirements

**Python packages:**
```
torch-geometric
mlflow
neuralforecast
prophet
wandb               # free tier
```

**Compute:**
- Colab/Kaggle for initial training (free GPU)
- Oracle Cloud for inference

---

### Phase 4 Requirements

**Python packages:**
```
langgraph
langchain-core
ray                 # local mode, no cluster needed
```

---

### Phase 5 Requirements

**Node packages:**
```
next
three
d3
zustand
tailwindcss
```

---

### Phase 6 Requirements

**DevOps:**
```
terraform
kubectl / k3s
helm
prometheus
grafana
locust              # load testing
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- Node.js 20+
- Git

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ai-digital-twin.git
cd ai-digital-twin

# 2. Copy and fill environment variables
cp .env.example .env
# Edit .env with your free API keys (Reddit, YouTube)

# 3. Start infrastructure services
docker compose up -d

# 4. Install Python dependencies
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 5. Seed initial graph data
python scripts/seed_graph.py

# 6. Start the API server
uvicorn api.main:app --reload

# 7. Start the frontend
cd frontend
npm install
npm run dev
```

Access:
- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:3000`
- Neo4j Browser: `http://localhost:7474`
- Grafana: `http://localhost:3001`
- MLflow: `http://localhost:5000`

---

## 📡 Data Sources (Free Tier / Open)

| Source | API / Method | Rate Limit | Notes |
|---|---|---|---|
| Reddit | PRAW (Official API) | 100 req/min | Free, requires app registration |
| Hacker News | Algolia HN API | Unlimited | Completely free |
| GDELT | Direct download / API | Unlimited | Fully open |
| RSS/News | feedparser | Unlimited | Any public RSS |
| YouTube | Data API v3 | 10K units/day | Free with Google account |
| Wikipedia | Wikimedia API | Unlimited | Free, no key (recent-changes stream) |
| Bluesky | AT Protocol AppView | Generous | Free, no key (app password optional for search) |

**No paid APIs required.**

---

## 🤖 ML Models (Open-Source Only)

| Task | Model | Source | Size |
|---|---|---|---|
| Text embeddings | all-MiniLM-L6-v2 | HuggingFace | 80MB |
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment | HuggingFace | 500MB |
| Topic modeling | BERTopic | PyPI | Library |
| Toxicity | Detoxify | PyPI | ~250MB |
| Summarization | facebook/bart-large-cnn | HuggingFace | 1.6GB |
| Stance detection | cross-encoder/nli-deberta-v3-small | HuggingFace | 180MB |
| Graph ML | PyG (custom GNN) | PyPI | Library |
| Time-series | NeuralForecast / Prophet | PyPI | Library |
| LLM (optional) | Llama 3.2 3B via Ollama | Ollama | 2GB |

All models run locally or on free compute (Colab/Kaggle for training, CPU inference in production for small models).

---

## 🔐 Environment Variables

```bash
# .env.example

# Reddit (free at reddit.com/prefs/apps)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=ai-digital-twin/1.0

# YouTube (free at console.cloud.google.com)
YOUTUBE_API_KEY=

# Neo4j (self-hosted)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

# PostgreSQL (Supabase free tier or local)
DATABASE_URL=postgresql://user:password@localhost:5432/digital_twin

# Redis (self-hosted)
REDIS_URL=redis://localhost:6379

# MLflow (self-hosted)
MLFLOW_TRACKING_URI=http://localhost:5000

# Groq (optional, free tier for LLM)
GROQ_API_KEY=

# App
APP_ENV=development
SECRET_KEY=change-this-in-production
```

---

## 🧪 Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests (requires Docker services running)
pytest tests/integration/

# With coverage
pytest --cov=. --cov-report=html
```

---

## 📊 Scalability Targets

| Metric | Phase 3 Target | Phase 6 Target |
|---|---|---|
| Event throughput | 1K events/min | 50K events/min |
| Graph nodes | 500K | 10M+ |
| Inference latency | <2s | <500ms |
| Concurrent simulations | 5 | 50+ |
| Real-time delay | <30s | <10s |

All targets achievable on Oracle Cloud Free Tier (4 OCPU, 24GB RAM) for reasonable workloads. Scale-out by adding free-tier nodes.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Write tests for your changes
4. Ensure all tests pass: `pytest`
5. Commit with conventional commits: `git commit -m "feat: add virality scorer"`
6. Push and open a Pull Request

See `docs/contributing.md` for full guidelines.

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🏆 What This Demonstrates

This project is not a typical ML portfolio project. It shows:

- **Distributed systems engineering** at internet scale
- **Real-time streaming** with Kafka and async Python
- **Graph machine learning** with PyG temporal GNNs
- **Multi-agent simulation** with LangGraph and Ray
- **Production MLOps** with MLflow, automated retraining, drift monitoring
- **Full-stack AI deployment** from data ingestion to 3D visualization
- **Zero-cost engineering** — knowing how to build serious systems without blowing budgets

> "I can engineer internet-scale AI systems — and I can do it for free."

---

*Built with ❤️ and $0.*
