"""Burst-ranking evaluation — the metric that actually has signal.

Instead of regressing the (noise-dominated) 24h engagement count, we ask a
ranking question: among a narrative's events, can the model rank the ones
followed by an above-baseline *surge* higher than the quiet ones?

For each event we compute the within-narrative residual of true log-engagement
(truth minus that narrative's mean) and label the top tercile as a "surge".
Each model scores events by its own within-narrative residual prediction. We
then report ROC-AUC, Average Precision (AP), Precision@K (K = #positives), and
within-narrative Spearman. Random = AUC 0.5 / AP = positive rate.

An ORACLE row (a leak-free recent-excitation feature) gives the achievable
ceiling on this synthetic data.

Reads the per-event dumps written by the models:
  results/simcity_test_preds.npz, results/vanilla_tgn_test_preds.npz,
  results/static_gnn_test_preds.npz
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from analysis_tools.oracle_residual_check import recent_excitation_features

MODELS = {
    "SimCity (full)": "results/simcity_test_preds.npz",
    "Vanilla TGN": "results/vanilla_tgn_test_preds.npz",
    "Static GNN + SEIR": "results/static_gnn_test_preds.npz",
}
SURGE_QUANTILE = 0.66  # top tercile of within-narrative residual = "surge"


def within_resid(values, groups):
    out = np.zeros_like(values, dtype=np.float64)
    for g in np.unique(groups):
        idx = groups == g
        out[idx] = values[idx] - values[idx].mean()
    return out


def rank_metrics(true, pred, dst):
    """Compute surge-ranking metrics from per-event (true, pred, dst)."""
    multi = pd.Series(dst).groupby(dst).transform("count").to_numpy() > 1
    t, p, d = true[multi], pred[multi], dst[multi]
    rt = within_resid(t, d)
    rp = within_resid(p, d)

    thr = np.quantile(rt, SURGE_QUANTILE)
    label = (rt >= thr).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return None

    auc = float(roc_auc_score(label, rp))
    ap = float(average_precision_score(label, rp))
    k = int(label.sum())
    topk = np.argsort(-rp)[:k]
    prec_at_k = float(label[topk].mean())
    # Spearman
    from scipy.stats import spearmanr
    rho, _ = spearmanr(rp, rt)
    return {
        "auc": auc,
        "avg_precision": ap,
        "precision_at_k": prec_at_k,
        "positive_rate": float(label.mean()),
        "spearman": float(rho),
        "n": int(len(label)),
    }


def oracle_row():
    df = pd.read_pickle("data/synthetic_events.pkl").sort_values("t").reset_index(drop=True)
    t = df["t"].to_numpy(float)
    y = df["log_engagement"].to_numpy(float)
    dst = df["dst"].to_numpy(np.int64)
    X = np.log1p(np.clip(recent_excitation_features(df), 0, None))
    tmin, tmax = t.min(), t.max()
    test = t >= tmin + 0.80 * (tmax - tmin)
    # score = the strongest recent-excitation channel (shortest decay, same-narr)
    return rank_metrics(y[test], X[test, 0], dst[test])


def main():
    rows = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            d = np.load(path)
            m = rank_metrics(d["true"].astype(float), d["pred"].astype(float),
                             d["dst"].astype(np.int64))
            if m:
                rows[name] = m
    orc = oracle_row()
    if orc:
        rows["Oracle (recent-excitation)"] = orc

    os.makedirs("results", exist_ok=True)
    with open("results/burst_ranking.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Burst-ranking evaluation (within-narrative surge prediction)",
        "",
        f"Surge = top {int((1-SURGE_QUANTILE)*100)}% of within-narrative residual "
        "engagement. Random AUC = 0.50; random AP = positive rate "
        f"(~{1-SURGE_QUANTILE:.2f}).",
        "",
        "| Model | ROC-AUC ^ | Avg Precision ^ | Precision@K ^ | Spearman ^ |",
        "|---|---|---|---|---|",
    ]
    order = ["SimCity (full)", "Vanilla TGN", "Static GNN + SEIR", "Oracle (recent-excitation)"]
    for name in order:
        if name in rows:
            m = rows[name]
            lines.append(
                f"| {name} | {m['auc']:.3f} | {m['avg_precision']:.3f} | "
                f"{m['precision_at_k']:.3f} | {m['spearman']:+.3f} |"
            )
    lines.append("")
    if rows:
        pr = next(iter(rows.values()))["positive_rate"]
        lines.append(f"> Positive rate = {pr:.3f} (random-baseline AP). AUC > 0.5 = captures surge signal.")
    md = "\n".join(lines) + "\n"
    with open("results/burst_ranking.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote results/burst_ranking.md and results/burst_ranking.json")


if __name__ == "__main__":
    main()
