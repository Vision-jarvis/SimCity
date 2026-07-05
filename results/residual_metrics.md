# Temporal-signal evaluation (within-narrative residuals)

Isolates the time-varying, excitation-driven component of engagement by removing each narrative's mean. `skill_vs_mean` > 0 means the model beats a narrative-mean predictor; Spearman measures within-narrative surge ranking.

| Model | Raw MAE | Residual MAE ↓ | Skill vs mean ↑ | Within-narr. Spearman ↑ |
|---|---|---|---|---|
| SimCity (full) | 0.8660 | 0.6337 | -0.0802 | -0.1194 |
| Vanilla TGN | 0.9273 | 0.6374 | -0.0866 | -0.0508 |
| Static GNN + SEIR | 0.9232 | 0.5886 | -0.0035 | -0.0027 |

> Narrative-mean predictor residual MAE = 0.5866 (skill = 0 baseline).
