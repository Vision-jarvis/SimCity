"""Re-derive the real corpus from the cached raw store with corrected clustering.

Why this exists: the original greedy clusterers capped the cluster count and
then force-assigned every remaining item to its nearest centroid *regardless of
similarity*, so most "narratives" in the real corpus were artifacts (only 2-3%
of cross-source pairs exceeded 0.5 title cosine). This script rebuilds the
dataset from the already-fetched raw store using the fixed, time-windowed
embedding clusterer -- no network access, no refetch.

It also upgrades content features from lexical hashing to real sentence
embeddings (revision Step 7).

Usage:
    python scripts/rederive_real_corpus.py --threshold 0.60 \
        --raw data/real_events_raw.pkl --out data/real_events_emb.pkl
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.build_real_dataset import (cluster_narratives_embedding,  # noqa: E402
                                     encode_titles)
from data.synthetic_generator import compute_future_engagement  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/real_events_raw.pkl")
    ap.add_argument("--out", default="data/real_events_emb.pkl")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--window-days", type=float, default=7.0)
    ap.add_argument("--emb-cache", default=None,
                    help="optional .npy cache of title embeddings")
    args = ap.parse_args()

    df = pd.read_pickle(args.raw).sort_values("t").reset_index(drop=True)
    print(f"raw store: {len(df)} events")

    if args.emb_cache and os.path.exists(args.emb_cache):
        emb = np.load(args.emb_cache)
        print(f"loaded cached embeddings {emb.shape}")
    else:
        print("encoding titles (sentence-transformers)...")
        emb = encode_titles(df["title"].tolist())
        if args.emb_cache:
            np.save(args.emb_cache, emb)
    assert len(emb) == len(df)

    times = df["t"].to_numpy(float)
    assign, n_clusters = cluster_narratives_embedding(
        emb, times=times, threshold=args.threshold, window_days=args.window_days)
    print(f"clusters: {n_clusters} (threshold {args.threshold}, "
          f"window {args.window_days}d)")

    # platform: 0 = HN (has points/url), 1 = news. The raw store keeps it if the
    # collector wrote it; otherwise infer from the HN-only 'points' field.
    if "platform" in df.columns:
        plat = df["platform"].to_numpy(int)
    else:
        plat = np.where(df["points"].to_numpy() > 0, 0, 1)

    authors = {a: i for i, a in enumerate(df["author"].unique())}
    num_users = len(authors)

    out = pd.DataFrame({
        "src": df["author"].map(authors).astype(int).to_numpy(),
        "dst": (assign + num_users).astype(int),
        "t": times,
        "platform": plat,
    })
    bin_s = 15 * 60
    b = ((out["t"] - out["t"].min()) // bin_s).astype(int)
    out["gdelt_volume"] = b.map(b.value_counts()).astype(float).values
    out["msg"] = list(emb.astype(np.float32))     # real semantic features
    out["log_engagement"] = compute_future_engagement(
        out, prediction_window_seconds=86400)

    span = out.groupby("dst")["platform"].nunique()
    print(f"  users/domains: {num_users} | narratives: {n_clusters} | events: {len(out)}")
    print(f"  platform distribution:\n{out['platform'].value_counts().sort_index()}")
    print(f"  cross-platform narratives: {int((span > 1).sum())} / {n_clusters}")

    y = out["log_engagement"].to_numpy()
    grand = y.mean()
    w = bt = 0.0
    for _, g in out.groupby("dst"):
        yi = g["log_engagement"].to_numpy()
        w += ((yi - yi.mean()) ** 2).sum()
        bt += len(yi) * (yi.mean() - grand) ** 2
    print(f"  within-narrative variance fraction: {w / max(w + bt, 1e-9):.3f}")

    out.to_pickle(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
