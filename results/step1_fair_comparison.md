# Step 1 (fair): per-narrative MHP baseline vs. SimCity bridge score

Both sides get exactly one validation-selected degree of freedom: the baseline picks its statistic **and** sign on validation; SimCity picks its head orientation on validation. Both are then evaluated once on test.

| Seed | Baseline statistic (val-picked) | Baseline test AUC | SimCity test AUC | Difference (95% CI) |
|---|---|---|---|---|
| 1 | rate | 0.631 | 0.882 | +0.250 [+0.172, +0.332] |
| 2 | rate | 0.631 | 0.673 | +0.042 [-0.029, +0.117] |
| 3 | rate | 0.631 | 0.779 | +0.147 [+0.047, +0.251] |

**Mean over seeds:** SimCity 0.778 ± 0.104 vs. per-narrative baseline 0.631 ± 0.000.

**Oracle upper bound (not a fair comparator):** if the baseline is allowed to pick its statistic on the *test* labels it reaches 0.728 (`mu`) --- within SimCity's seed spread. We report this so the margin is not overstated: SimCity's advantage over per-narrative statistics is real under a matched protocol but is not large.

> Verdict: SimCity > per-narrative baseline.
