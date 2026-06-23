# API Reference

## Base URL
```
http://localhost:8000
```

## Endpoints

### Health & Discovery
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API root with endpoint listing |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | ReDoc API documentation |
| POST | `/graphql` | GraphQL endpoint |

### Graph (`/graph`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/graph/nodes?label=&limit=100` | Get graph nodes |
| GET | `/graph/edges?type=&limit=100` | Get graph edges |
| GET | `/graph/stats` | Graph statistics |
| GET | `/graph/communities` | Detected communities |
| GET | `/graph/influence?limit=20` | Top influencers |

### Simulation (`/simulate`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/simulate/run` | Run SEIR-Z-D simulation |
| POST | `/simulate/intervention` | Counterfactual what-if: baseline vs. intervention deltas |
| GET | `/simulate/replay/{run_id}` | Replay cached results |
| GET | `/simulate/presets` | List scenario presets |

### Trends (`/trends`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/trends/current?limit=20` | Current trending topics |
| POST | `/trends/forecast` | Forecast engagement |
| GET | `/trends/narratives?limit=10` | Active narratives |
| POST | `/trends/narrative-transfer` | Cross-platform narrative-transfer detection |

### Search (`/search`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/search/semantic` | Semantic search |
| GET | `/search/nlp/analyze?text=` | NLP analysis |

### WebSocket
| Protocol | Path | Description |
|----------|------|-------------|
| WS | `/ws/stream` | Real-time event stream |
| WS | `/ws/simulation/{run_id}` | Simulation playback |

## Request/Response Examples

### Run Simulation
```json
POST /simulate/run
{
  "scenario_id": "misinfo_outbreak",
  "N": 100000,
  "initial_S": 90000,
  "initial_E": 1000,
  "steps": 30,
  "theta": 2.5,
  "sigma": 0.7
}
```

### Counterfactual Intervention
```json
POST /simulate/intervention
{
  "scenario": { "N": 100000, "steps": 40, "theta": 2.5 },
  "interventions": [
    { "type": "fact_check", "start_step": 10, "magnitude": 0.7 }
  ],
  "include_history": true
}
```
Response includes `baseline_metrics`, `treatment_metrics`, `deltas`, and
`pct_change` (e.g. peak infection, total reach, persistent zealots).

### Forecast
```json
POST /trends/forecast
{
  "topic": "AI Safety",
  "horizon_hours": 24
}
```

### Narrative Transfer Detection
```json
POST /trends/narrative-transfer
{
  "events": [
    {"content": "New AI safety regulation proposed", "platform": 0, "timestamp": 1700000000},
    {"content": "AI safety regulation bill sparks debate", "platform": 1, "timestamp": 1700003600},
    {"content": "AI regulation covered by news outlets", "platform": 2, "timestamp": 1700007200}
  ],
  "similarity_threshold": 0.25,
  "transfers_only": true
}
```
Returns each narrative's `transfer_path` (ordered platform hops with `lag_hours`),
`mutation_score` (content drift across platforms), and `virality_score`.

### Semantic Search
```json
POST /search/semantic
{
  "query": "artificial intelligence regulation",
  "platform": "reddit",
  "limit": 10
}
```
