# Burst-ranking evaluation (within-narrative surge prediction)

Surge = top 34% of within-narrative residual engagement. Random AUC = 0.50; random AP = positive rate (~0.34).

| Model | ROC-AUC ^ | Avg Precision ^ | Precision@K ^ | Spearman ^ |
|---|---|---|---|---|
| SimCity (full) | 0.409 | 0.282 | 0.254 | -0.167 |
| Vanilla TGN | 0.454 | 0.302 | 0.299 | -0.090 |
| Static GNN + SEIR | 0.497 | 0.342 | 0.351 | +0.001 |
| Oracle (recent-excitation) | 0.507 | 0.361 | 0.371 | +0.141 |

> Positive rate = 0.340 (random-baseline AP). AUC > 0.5 = captures surge signal.
