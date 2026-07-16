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

## Fix applied: validation-split orientation — real-data validation ACHIEVED

Fresh 3-seed run with the fix (`train.py` dumps val predictions;
`narrative_transfer_eval.py --orient-with`):

| seed | val AUC → orientation | raw test | **oriented test** | p | count-strata |
|---|---|---|---|---|---|
| 1 | 0.245 → flip | 0.118 | **0.882** | 1.0e-24 | 0.894 |
| 2 | 0.647 → keep | 0.673 | **0.673** | 1.2e-08 | 0.685 |
| 3 | 0.261 → flip | 0.221 | **0.779** | 4.9e-14 | 0.842 |

**Oriented 3-seed AUC: 0.778 ± 0.104.** Validation orientation predicted the
correct test-side orientation in 3/3 seeds. The real-data transfer signal is
stronger than the synthetic benchmark's 0.653. Raw values retained above and in
`results/narrative_transfer_real_oriented.json` for transparency.

History: a 93/7 imbalanced snapshot gave chance for every seed; at 71 transfer
cases (12.5k events) no cross-seed structure was detectable. Corpus scale
revealed the structure. Per-seed raw values:
`results/narrative_transfer_real_seeds.json`.
