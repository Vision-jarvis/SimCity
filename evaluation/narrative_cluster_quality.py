"""Step 6/7: how good are the narrative clusters actually?

The transfer task is only meaningful if a "narrative" that spans Hacker News
and GDELT really is the *same* story. This script:

  1. Samples cross-source narrative pairs (one HN title + one news title from
     the same cluster) and writes a CSV for **human labelling** --- the column
     `same_event` is deliberately left blank. Cluster precision reported in the
     paper must come from those human labels, not from a model.
  2. Computes an automated *proxy* using sentence-embedding cosine similarity,
     clearly reported as a proxy and used only to compare clustering methods
     (token-Jaccard vs. embedding) on equal footing.

Usage:
    python -m evaluation.narrative_cluster_quality --data data/real_events.pkl \
        --raw data/real_events_raw.pkl --n 100
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def sample_cross_source_pairs(df_raw, assign, n_pairs, rng):
    """Sample (HN title, news title) pairs drawn from the same cluster."""
    df = df_raw.copy()
    df["cluster"] = assign
    pairs = []
    for cid, g in df.groupby("cluster"):
        hn = g[g["platform"] == 0]
        nw = g[g["platform"] == 1]
        if len(hn) == 0 or len(nw) == 0:
            continue
        k = min(3, len(hn), len(nw))
        for _ in range(k):
            a = hn.iloc[rng.integers(0, len(hn))]
            b = nw.iloc[rng.integers(0, len(nw))]
            pairs.append({"cluster": int(cid),
                          "hn_title": a["title"], "news_title": b["title"],
                          "news_domain": b["author"]})
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/real_events_raw.pkl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--emb-threshold", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from data.build_real_dataset import (cluster_narratives,
                                         cluster_narratives_embedding,
                                         encode_titles)

    df = pd.read_pickle(args.raw).sort_values("t").reset_index(drop=True)
    titles = df["title"].tolist()
    print(f"raw store: {len(df)} events")

    print("clustering (token-Jaccard)...")
    jac_assign, n_jac = cluster_narratives(titles)
    print("encoding titles + clustering (sentence embeddings)...")
    emb = encode_titles(titles)
    emb_assign, n_emb = cluster_narratives_embedding(emb, threshold=args.emb_threshold)

    rng = np.random.default_rng(args.seed)
    out = {}
    for name, assign in (("jaccard", jac_assign), ("embedding", emb_assign)):
        pairs = sample_cross_source_pairs(df, assign, args.n, rng)
        # automated PROXY: cosine similarity between the paired titles
        if pairs:
            idx = {t: i for i, t in enumerate(titles)}
            sims = []
            for p in pairs:
                i, j = idx.get(p["hn_title"]), idx.get(p["news_title"])
                if i is not None and j is not None:
                    sims.append(float(emb[i] @ emb[j]))
            sims = np.array(sims) if sims else np.array([0.0])
            out[name] = {
                "n_clusters": int(n_jac if name == "jaccard" else n_emb),
                "n_cross_source_pairs_sampled": len(pairs),
                "proxy_mean_cosine": float(sims.mean()),
                "proxy_frac_above_0.5": float((sims > 0.5).mean()),
                "proxy_frac_above_0.6": float((sims > 0.6).mean()),
            }
            csv = pd.DataFrame(pairs)
            csv["cosine_proxy"] = list(sims) + [np.nan] * (len(csv) - len(sims))
            csv["same_event"] = ""     # <-- human labels go here
            path = f"results/cluster_labelling_{name}.csv"
            csv.to_csv(path, index=False, encoding="utf-8")
            print(f"  {name}: {len(pairs)} pairs -> {path}")

    os.makedirs("results", exist_ok=True)
    with open("results/narrative_cluster_quality.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = [
        "# Narrative cluster quality (revision Steps 6 & 7)",
        "",
        "Cross-source pairs (one HN title + one news title from the same "
        "narrative cluster) sampled for **human labelling**; see "
        "`results/cluster_labelling_*.csv`, column `same_event` (blank by "
        "design). The cosine figures below are an automated **proxy** used only "
        "to compare clustering methods --- they are not a substitute for the "
        "human precision number the paper should report.",
        "",
        "| Clustering | Clusters | Pairs sampled | Proxy mean cosine | Proxy frac > 0.5 |",
        "|---|---|---|---|---|",
    ]
    for k, v in out.items():
        lines.append(f"| {k} | {v['n_clusters']} | {v['n_cross_source_pairs_sampled']} "
                     f"| {v['proxy_mean_cosine']:.3f} | {v['proxy_frac_above_0.5']:.2f} |")
    lines.append("")
    with open("results/narrative_cluster_quality.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
