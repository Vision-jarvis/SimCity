# Multi-seed next-platform prediction (mean +/- std over seeds)

Data: `data/synthetic_events.pkl` | seeds: [1, 2, 3]

| Model | AUC | Top-1 acc | Log-loss |
|---|---|---|---|
| SimCity (Hawkes-weighted) | 0.610 ± 0.005 | 0.419 ± 0.009 | 1.079 |
| Static MHP | 0.605 ± 0.000 | 0.416 ± 0.000 | 1.077 |

Welch t-test on seed AUCs (SimCity vs Static): t = 1.72, p = 0.2270
Per-seed SimCity AUC: [0.605, 0.614, 0.611]
Per-seed Static AUC:  [0.605, 0.605, 0.605]
