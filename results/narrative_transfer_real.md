# Narrative transfer detection (which narratives jump platforms?)

Test narratives: 257 | transferred: 53 (20.6%). Scores computed only from pre-switch (single-platform) events — causal.

| Scorer | AUC |
|---|---|
| SimCity bridge score (per-narrative off-diag alpha/gamma) | 0.426 |
| Popularity baseline (pre-switch event count) | 0.648 |
| SimCity bridge, within count-matched strata | 0.570 |
| Static MHP (single global alpha) | 0.500 (by construction) |

> A static Hawkes cannot rank narratives at all; any AUC > 0.5 here is capability the neural parameterisation adds. The count-stratified row controls for the popularity confound.
