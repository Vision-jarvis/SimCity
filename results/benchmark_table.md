# SimCity Benchmark Results

Dataset: `data/synthetic_events.pkl`  (8203 events, chronological 70/10/20 split)

## Virality regression (log-engagement, held-out test)

| Model | MAE ↓ | RMSE ↓ |
|---|---|---|
| Naive train-mean | 0.8246 | 0.9762 |
| Static GNN + SEIR | 0.9232 | 1.1539 |
| Vanilla TGN | 0.9273 | 1.1398 |
| **SimCity (full)** | 0.8660 | 1.0410 |

## Cross-platform burst timing (Hawkes NLL, held-out test)

| Model | NLL ↓ |
|---|---|
| Static MHP baseline | 0.7029 |
| **SimCity (full)** | 0.7343 |

> Note: numbers above are on **synthetic** multiplex data. Replace with the real ingested Reddit/HN/GDELT dataset before paper submission.
