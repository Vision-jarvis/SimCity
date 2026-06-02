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
| GET | `/simulate/replay/{run_id}` | Replay cached results |
| GET | `/simulate/presets` | List scenario presets |

### Trends (`/trends`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/trends/current?limit=20` | Current trending topics |
| POST | `/trends/forecast` | Forecast engagement |
| GET | `/trends/narratives?limit=10` | Active narratives |

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

### Forecast
```json
POST /trends/forecast
{
  "topic": "AI Safety",
  "horizon_hours": 24
}
```

### Semantic Search
```json
POST /search/semantic
{
  "query": "artificial intelligence regulation",
  "platform": "reddit",
  "limit": 10
}
```
