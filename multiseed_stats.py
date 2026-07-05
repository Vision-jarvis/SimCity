"""Multi-seed statistics for the headline timing result (defensibility pass).

Aggregates next-platform prediction AUC over N seeds for both SimCity
(seed-tagged intensity dumps from train.py) and the static MHP (retrained here
per seed — cheap). Reports mean +/- std, per-seed values, and a Welch t-test on
the seed-level AUCs.

Usage (after the seed runs finish):
    python multiseed_stats.py --data data/synthetic_events.pkl --seeds 1 2 3
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

from platform_prediction_eval import _metrics, train_static_mhp_intensities


def simcity_aucs(seeds):
    out = {}
    for s in seeds:
        path = f"results/simcity_test_preds_seed{s}_hawkes.npz"
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        d = np.load(path)
        m = _metrics(d["intensity"].astype(float), d["platform"].astype(int))
        out[s] = m
    return out


def static_aucs(data, seeds):
    out = {}
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        inten, plat, _, _ = train_static_mhp_intensities(data)
        out[s] = _metrics(inten, plat)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_events.pkl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    sc = simcity_aucs(args.seeds)
    st = static_aucs(args.data, args.seeds)

    def arr(d, k):
        return np.array([v[k] for v in d.values()])

    res = {"seeds": args.seeds}
    lines = [
        "# Multi-seed next-platform prediction (mean +/- std over seeds)",
        "",
        f"Data: `{args.data}` | seeds: {args.seeds}",
        "",
        "| Model | AUC | Top-1 acc | Log-loss |",
        "|---|---|---|---|",
    ]
    for name, d in [("SimCity (Hawkes-weighted)", sc), ("Static MHP", st)]:
        if not d:
            continue
        auc, acc, ll = arr(d, "macro_auc"), arr(d, "top1_acc"), arr(d, "log_loss")
        res[name] = {
            "auc_per_seed": auc.tolist(), "auc_mean": float(auc.mean()), "auc_std": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
            "acc_mean": float(acc.mean()), "logloss_mean": float(ll.mean()),
        }
        lines.append(
            f"| {name} | {auc.mean():.3f} ± {auc.std(ddof=1) if len(auc)>1 else 0:.3f} "
            f"| {acc.mean():.3f} ± {acc.std(ddof=1) if len(acc)>1 else 0:.3f} "
            f"| {ll.mean():.3f} |"
        )

    if sc and st and len(sc) > 1:
        a, b = arr(sc, "macro_auc"), arr(st, "macro_auc")
        tstat, pval = stats.ttest_ind(a, b, equal_var=False)
        res["welch_t"] = {"t": float(tstat), "p": float(pval)}
        lines += ["", f"Welch t-test on seed AUCs (SimCity vs Static): t = {tstat:.2f}, p = {pval:.4f}"]
        lines += [f"Per-seed SimCity AUC: {[round(x,3) for x in a]}",
                  f"Per-seed Static AUC:  {[round(x,3) for x in b]}"]

    os.makedirs("results", exist_ok=True)
    with open("results/multiseed_stats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    with open("results/multiseed_stats.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
