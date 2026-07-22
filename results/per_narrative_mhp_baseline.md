# Per-narrative MHP baseline (revision Step 1)

Test narratives: 259 (166 transfer). Per-narrative Hawkes MLE converged for 130.

**Structural note:** a narrative's pre-switch window contains events on a *single* platform by construction, so per-narrative fitting cannot identify cross-platform alpha at all. It identifies the narrative's own excitation dynamics. Scores below are the most generous reading of that baseline (best sign taken for each statistic).

| Scorer | AUC (best sign) | count-stratified |
|---|---|---|
| per-narrative: mu | 0.728 | 0.682 |
| per-narrative: rate | 0.631 | 0.707 |
| per-narrative: alpha | 0.583 | 0.551 |
| per-narrative: n_events | 0.576 | 0.545 |
| per-narrative: branching | 0.569 | 0.545 |
| per-narrative: ia_cv | 0.567 | 0.532 |
| per-narrative: burstiness | 0.522 | 0.532 |
| per-narrative: gamma | 0.512 | 0.496 |
| **SimCity bridge score** | **0.673** | 0.685 |

> Best per-narrative statistic: **mu** (AUC 0.728). Verdict: per-narrative baseline matches or beats SimCity.
