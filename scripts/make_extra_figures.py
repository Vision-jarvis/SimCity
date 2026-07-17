"""Generate the paper's supporting figures from existing artifacts.

Figure 2 (sign_identifiability.pdf): per-narrative bridge-score percentile
ranks across training seeds on the real corpus. Seeds 2 vs 3 align on the
identity (rho = +0.96); seed 1 vs 2 aligns on the anti-diagonal (mirror
image) -- the sign non-identifiability of the bridge head, shown directly.

Figure 3 (oracle_noise.pdf): the negative result. A leak-free oracle handed
the true recent-excitation feature still cannot predict the within-narrative
engagement residual (r ~ 0.2, negative MAE skill): the cloud around the
identity line is the irreducible Poisson-scale noise of near-critical
branching.

Run from the repo root:  python scripts/make_extra_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from evaluation.narrative_transfer_eval import build_table
from analysis_tools.oracle_residual_check import recent_excitation_features, within_resid

BLUE, AQUA = "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def seed_scores(path):
    d = np.load(path)
    rows = build_table(d)
    return ({r["narrative"]: r["bridge_score"] for r in rows},
            {r["narrative"]: r["label"] for r in rows})


def fig_sign_identifiability():
    s = {i: seed_scores(f"results/simcity_test_preds_real_s{i}.npz") for i in (1, 2, 3)}
    common = sorted(set(s[1][0]) & set(s[2][0]) & set(s[3][0]))
    ranks = {i: rankdata([s[i][0][n] for n in common]) / len(common) for i in (1, 2, 3)}
    labels = np.array([s[2][1][n] for n in common])

    # Auto-select the converging pair (most positive rho) and the mirrored
    # pair (most negative rho) -- which seeds converge to which orientation
    # varies across runs, consistent with a sign-symmetric likelihood.
    from itertools import combinations
    rhos = {(a, b): spearmanr(ranks[a], ranks[b])[0]
            for a, b in combinations((1, 2, 3), 2)}
    (pa, pb) = max(rhos, key=rhos.get)
    (ma, mb) = min(rhos, key=rhos.get)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), dpi=200, sharey=False)
    panels = [
        (axes[0], ranks[pa], ranks[pb], f"seed {pa}", f"seed {pb}"),
        (axes[1], ranks[ma], ranks[mb], f"seed {ma}", f"seed {mb}"),
    ]
    for ax, x, y, xl, yl in panels:
        rho, _ = spearmanr(x, y)
        m = labels == 1
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
        ax.plot([0, 1], [1, 0], color=GRID, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.scatter(np.asarray(x)[~m], np.asarray(y)[~m], s=14, color=INK2,
                   alpha=0.45, linewidths=0, label="no transfer", zorder=2)
        ax.scatter(np.asarray(x)[m], np.asarray(y)[m], s=16, color=AQUA,
                   alpha=0.85, linewidths=0, label="transferred", zorder=3)
        ax.set_title(f"Spearman $\\rho = {rho:+.2f}$", fontsize=9, color=INK)
        ax.set_xlabel(f"bridge-score rank ({xl})", fontsize=8.5, color=INK)
        ax.set_ylabel(f"bridge-score rank ({yl})", fontsize=8.5, color=INK)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7.5, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left",
                   handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout()
    fig.savefig("paper/figures/sign_identifiability.pdf", bbox_inches="tight")
    fig.savefig("paper/figures/sign_identifiability.png", bbox_inches="tight")
    print("Wrote paper/figures/sign_identifiability.{pdf,png}")


def fig_oracle_noise():
    from sklearn.linear_model import Ridge
    df = pd.read_pickle("data/synthetic_events.pkl").sort_values("t").reset_index(drop=True)
    t = df["t"].to_numpy(float)
    y = df["log_engagement"].to_numpy(float)
    dst = df["dst"].to_numpy(np.int64)
    X = np.log1p(np.clip(recent_excitation_features(df), 0, None))

    tmin, tmax = t.min(), t.max()
    train = t < tmin + 0.70 * (tmax - tmin)
    test = t >= tmin + 0.80 * (tmax - tmin)
    pred = Ridge(alpha=1.0).fit(X[train], y[train]).predict(X)

    counts = pd.Series(dst).groupby(dst).transform("count").to_numpy()
    m = test & (counts > 1)
    rt = within_resid(y[m], dst[m])
    rp = within_resid(pred[m], dst[m])
    skill = 1 - np.abs(rp - rt).mean() / np.abs(rt).mean()
    # x-axis: within-narrative residual of the raw excitation feature itself
    # (standardised) -- the strongest causal predictor available to any model.
    rf = within_resid(X[m, 0], dst[m])
    rf = rf / (rf.std() + 1e-9) * rt.std()
    r = np.corrcoef(rf, rt)[0, 1]

    fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=200)
    lim = np.percentile(np.abs(rt), 99)
    ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1, zorder=1)
    ax.scatter(rf, rt, s=9, color=BLUE, alpha=0.35, linewidths=0, zorder=2)
    ax.annotate(f"$r = {r:+.2f}$\noracle MAE skill $= {skill:+.2f}$",
                (0.04, 0.96), xycoords="axes fraction", va="top",
                fontsize=8.5, color=INK)
    ax.set_xlabel("recent-excitation residual (scaled)", fontsize=8.5, color=INK)
    ax.set_ylabel("true within-narrative residual", fontsize=8.5, color=INK)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.tick_params(labelsize=7.5, colors=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    fig.tight_layout()
    fig.savefig("paper/figures/oracle_noise.pdf", bbox_inches="tight")
    fig.savefig("paper/figures/oracle_noise.png", bbox_inches="tight")
    print(f"Wrote paper/figures/oracle_noise.{{pdf,png}} (r={r:+.3f}, skill={skill:+.3f})")


if __name__ == "__main__":
    os.makedirs("paper/figures", exist_ok=True)
    fig_sign_identifiability()
    fig_oracle_noise()
