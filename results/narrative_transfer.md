# Narrative transfer detection (which narratives jump platforms?)

Test narratives: 259 | transferred: 166 (64.1%). Scores computed only from pre-switch (single-platform) events, causal.

| Scorer | AUC |
|---|---|
| SimCity bridge score (per-narrative off-diag alpha/gamma) | 0.673 |
| Popularity baseline (pre-switch event count) | 0.424 |
| SimCity bridge, within count-matched strata | 0.685 |
| Static MHP (single global alpha) | 0.500 (by construction) |

> A static Hawkes cannot rank narratives at all; any AUC > 0.5 here is capability the neural parameterisation adds. The count-stratified row controls for the popularity confound.
