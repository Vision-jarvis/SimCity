# Cross-platform next-event prediction, SYNTHETIC (Hawkes-branching) data

Causal / leak-free. Random acc=0.333, AUC=0.5.

| Model | Top-1 acc | Macro AUC | Log-loss |
|---|---|---|---|
| SimCity (Hawkes-weighted) | 0.428 | 0.623 | 1.073 |
| SimCity (default weights) | 0.368 | 0.576 | 1.101 |
| Static MHP | 0.416 | 0.605 | 1.077 |
| Persistence | 0.372 | 0.529 | 1.948 |
| Marginal prior | 0.353 | 0.500 | 1.100 |

> On data with genuine cross-platform excitation, SimCity beats Static MHP once the Hawkes task is not diluted (AUC 0.623 vs 0.605).
