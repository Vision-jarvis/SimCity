# Narrative cluster quality (revision Steps 6 & 7)

Cross-source pairs (one HN title + one news title from the same narrative cluster) sampled for **human labelling**; see `results/cluster_labelling_*.csv`, column `same_event` (blank by design). The cosine figures below are an automated **proxy** used only to compare clustering methods --- they are not a substitute for the human precision number the paper should report.

| Clustering | Clusters | Pairs sampled | Proxy mean cosine | Proxy frac > 0.5 |
|---|---|---|---|---|
| jaccard | 400 | 100 | 0.128 | 0.03 |
| embedding | 400 | 100 | 0.215 | 0.02 |
