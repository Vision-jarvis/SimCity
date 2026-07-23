# What predicts real cross-platform transfer?

Corrected real corpus, 4904 test narratives (85 transfer, 1.7% base rate). Stratified 5-fold CV ROC-AUC, logistic regression on causal pre-switch features.

| Predictor | CV ROC-AUC |
|---|---|
| SimCity bridge score (neural Hawkes) | 0.513 (chance) |
| popularity only (n_events) | **0.652** ± 0.036 |
| engagement only (mean+max) | **0.814** ± 0.070 |
| all interpretable features | **0.823** ± 0.048 |

Univariate ROC-AUC per feature (each alone, best sign; collinearity-free), most predictive first:

- `mean_engag`: 0.815
- `max_engag`: 0.808
- `src_diversity`: 0.659
- `n_events`: 0.652
- `duration_h`: 0.636
- `log_rate`: 0.630
- `burstiness`: 0.557
- `mean_gdelt`: 0.538

> Real transfer is predicted by **early engagement** (mean/max log-engagement in the single-platform phase, AUC ~0.81 as a single feature), not by temporal-excitation structure (neural Hawkes at chance). Narratives that draw engagement early cross platforms; how their events cluster in time does not matter.
