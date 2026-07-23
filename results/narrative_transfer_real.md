# Narrative transfer detection, REAL HN+GDELT corpus

> NOTE: an earlier version of this file reported AUC 0.78 and a
> "validation-split orientation" success. That result was an artifact of a
> clustering bug (see `FINDINGS.md`). After correcting the clusterer and
> re-deriving the corpus over validated narratives, the finding below supersedes
> it. The paper reflects the corrected result.

Corrected corpus (`data/real_events_emb.pkl`, 14,169 validated narratives;
4,904 test narratives, 85 transfer):

| Scorer | 3-seed AUC |
|---|---|
| SimCity bridge score (neural Hawkes) | **0.51 +/- 0.02 (chance)** |
| Popularity baseline | 0.65 |
| Interpretable engagement model (5-fold CV) | **0.82** |

Per-seed bridge AUC: 0.532 / 0.497 / 0.511 (none significant,
p = 0.15 / 0.54 / 0.36).

Conclusion: on validated real narratives, narrative-conditioned Hawkes
excitation does NOT predict cross-platform transfer (chance). Real transfer is
predicted by early engagement, not temporal-excitation structure. Consistent
with the non-Hawkes SIR benchmark. See `results/real_transfer_predictors.md`
and `FINDINGS.md`.
