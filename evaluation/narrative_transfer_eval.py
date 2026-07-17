"""Narrative transfer detection — a claim ONLY the neural model can attempt.

Task: among narratives currently seen on a single platform, predict WHICH will
later jump to another platform. SimCity's virality head emits per-event,
narrative-conditioned Hawkes alpha matrices; its mean off-diagonal branching
mass ("bridge score") is a per-narrative transfer propensity. A static MHP has
one global alpha for all narratives, so it is structurally incapable of ranking
narratives — its AUC on this task is 0.5 by construction.

Causality: for each narrative we use only events BEFORE its first platform
switch (model memory at those events has seen a single-platform history);
label = whether a switch happens afterwards within the test window.

Confound control: popular narratives transfer more often, so we report the
event-count baseline AUC and the model's AUC within count-matched strata.

Usage:
    python narrative_transfer_eval.py [--preds results/simcity_test_preds.npz]
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def build_table(d):
    dst = d["dst"].astype(np.int64)
    plat = d["event_platform"].astype(np.int64)
    t = d["event_t"].astype(np.float64)
    bridge = d["alpha_off"].astype(np.float64)

    rows = []
    for g in np.unique(dst):
        idx = np.where(dst == g)[0]
        order = idx[np.argsort(t[idx])]
        p = plat[order]
        switch = np.where(p != p[0])[0]
        if switch.size:
            pre = order[: switch[0]]
            label = 1
        else:
            pre = order
            label = 0
        if pre.size == 0:
            continue
        rows.append({
            "narrative": int(g),
            "label": label,
            "bridge_score": float(bridge[pre].mean()),
            "n_pre_events": int(pre.size),
        })
    return rows


def stratified_auc(labels, scores, counts, n_bins=4):
    """AUC of `scores` within event-count strata (controls popularity confound)."""
    qs = np.quantile(counts, np.linspace(0, 1, n_bins + 1))
    aucs, weights = [], []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        m = (counts >= lo) & (counts <= hi if i == n_bins - 1 else counts < hi)
        if m.sum() >= 8 and 0 < labels[m].sum() < m.sum():
            aucs.append(roc_auc_score(labels[m], scores[m]))
            weights.append(m.sum())
    if not aucs:
        return float("nan")
    return float(np.average(aucs, weights=weights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/simcity_test_preds.npz")
    ap.add_argument("--out", default="results/narrative_transfer.md")
    ap.add_argument("--orient-with", default=None, metavar="VAL_NPZ",
                    help="Validation-prediction dump used to fix the bridge "
                         "head's sign (the Hawkes likelihood does not identify "
                         "its orientation). No test leakage: the sign is chosen "
                         "from validation labels only.")
    args = ap.parse_args()

    d = np.load(args.preds)
    if "alpha_off" not in d:
        raise SystemExit(f"{args.preds} has no alpha_off — rerun train.py (updated evaluate.py dumps it).")

    rows = build_table(d)
    labels = np.array([r["label"] for r in rows])
    bridge = np.array([r["bridge_score"] for r in rows])
    counts = np.array([r["n_pre_events"] for r in rows], dtype=float)

    orientation = 1.0
    if args.orient_with:
        vd = np.load(args.orient_with)
        vrows = build_table(vd)
        vlab = np.array([r["label"] for r in vrows])
        vsc = np.array([r["bridge_score"] for r in vrows])
        if 0 < vlab.sum() < len(vlab):
            val_auc = roc_auc_score(vlab, vsc)
            orientation = 1.0 if val_auc >= 0.5 else -1.0
            print(f"[orient] val AUC = {val_auc:.3f} -> orientation {orientation:+.0f}")
            bridge = orientation * bridge
        else:
            print("[orient] degenerate validation labels; keeping raw orientation")

    n, pos = len(rows), int(labels.sum())
    if pos == 0 or pos == n:
        raise SystemExit(f"Degenerate labels: {pos}/{n} positives — need more data.")

    auc_model = float(roc_auc_score(labels, bridge))
    auc_pop = float(roc_auc_score(labels, counts))
    auc_strat = stratified_auc(labels, bridge, counts)

    # Significance: Mann-Whitney U (equivalent to AUC) + bootstrap 95% CI.
    from scipy.stats import mannwhitneyu
    _, p_mw = mannwhitneyu(bridge[labels == 1], bridge[labels == 0], alternative="greater")
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(labels), len(labels))
        if 0 < labels[idx].sum() < len(idx):
            boots.append(roc_auc_score(labels[idx], bridge[idx]))
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    res = {
        "n_narratives": n, "n_transfer": pos,
        "orientation": orientation,
        "auc_bridge_score": auc_model,
        "auc_bridge_ci95": [ci_lo, ci_hi],
        "mannwhitney_p_one_sided": float(p_mw),
        "auc_popularity_baseline": auc_pop,
        "auc_bridge_within_count_strata": auc_strat,
        "auc_static_mhp": 0.5,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/narrative_transfer.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    md = [
        "# Narrative transfer detection (which narratives jump platforms?)",
        "",
        f"Test narratives: {n} | transferred: {pos} ({pos/n:.1%}). "
        "Scores computed only from pre-switch (single-platform) events — causal.",
        "",
        "| Scorer | AUC |",
        "|---|---|",
        f"| SimCity bridge score (per-narrative off-diag alpha/gamma) | {auc_model:.3f} |",
        f"| Popularity baseline (pre-switch event count) | {auc_pop:.3f} |",
        f"| SimCity bridge, within count-matched strata | {auc_strat:.3f} |",
        "| Static MHP (single global alpha) | 0.500 (by construction) |",
        "",
        "> A static Hawkes cannot rank narratives at all; any AUC > 0.5 here is "
        "capability the neural parameterisation adds. The count-stratified row "
        "controls for the popularity confound.",
        "",
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(json.dumps(res, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
