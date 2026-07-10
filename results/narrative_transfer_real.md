# Narrative transfer detection — REAL HN+GDELT corpus (multi-seed)

Balanced corpus: 12,574 events (46/54 HN/news), 188 test narratives, 71
transfer. Scores computed causally from pre-switch events only.

**The real-data bridge score is NOT seed-stable at this corpus size.**

| Scorer | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|
| SimCity bridge score | 0.319 | 0.632 | 0.248 | **0.400 ± 0.204** |
| — within count strata | 0.167 | 0.580 | 0.199 | unstable |
| Popularity baseline | 0.612 | 0.612 | 0.612 | deterministic |
| Static MHP | 0.500 | 0.500 | 0.500 | by construction |

Seed 2 alone reaches AUC 0.632 (Mann-Whitney p = 0.0012) and was briefly
reported as a real-data replication before the multi-seed check refuted it.
Contrast: the synthetic benchmark gives 0.653 ± 0.005 across the same seeds.

**Status:** transfer capability is established on the controlled benchmark
only. With 71 transfer cases the real-data score is dominated by training-seed
variance. Path forward: grow the corpus via the daily accumulation task
(`scripts/accumulate_real_data.cmd`) and re-run
`narrative_transfer_eval.py` per seed once transfer cases reach ~150+.

Per-seed raw values: `results/narrative_transfer_real_seeds.json`.
