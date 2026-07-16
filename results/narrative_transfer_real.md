# Narrative transfer detection — REAL HN+GDELT corpus (multi-seed)

Accumulated corpus: 32,472 events (40/60 HN/news), 259 test narratives,
**166 transfer cases** (power gate ≥150 passed). Scores computed causally from
pre-switch events only.

## Result: signal replicated across seeds — up to a SIGN ambiguity

| Scorer | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| SimCity bridge score | 0.276 | 0.590 (p=0.008) | 0.660 (p=1e-5) |
| — within count strata | 0.218 | 0.541 | 0.586 |
| Popularity baseline | 0.424 | 0.424 | 0.424 |
| Static MHP | 0.500 | 0.500 | 0.500 |

Cross-seed Spearman correlation of the per-narrative bridge scores:

| | s1↔s2 | s1↔s3 | s2↔s3 |
|---|---|---|---|
| ρ | −0.493 | −0.644 | **+0.957** |

**Diagnosis:** seeds 2 and 3 converge to essentially the *same* narrative
ranking (ρ=0.957) and both are individually significant; seed 1 learned the
*mirror image* of that ranking. The transfer signal is consistently
recoverable from real data — the Hawkes likelihood simply does not identify
the *orientation* of the narrative-conditioned bridge head (sign-symmetric
parameterizations fit equally well). Synthetic data (68% excitation variance)
breaks the symmetry; real data does not.

**Identified fix (next experiment):** orient each trained head on the
validation split (no test leakage) before evaluating. Requires adding a
validation-prediction dump to `train.py`.

History: a 93/7 imbalanced snapshot gave chance for every seed; at 71 transfer
cases (12.5k events) no cross-seed structure was detectable. Corpus scale
revealed the structure. Per-seed raw values:
`results/narrative_transfer_real_seeds.json`.
