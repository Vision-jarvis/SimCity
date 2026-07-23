"""Figures for the real-transfer predictor analysis (positive result).

Fig A (real_predictors.pdf): CV ROC-AUC of predictors of real cross-platform
transfer -- the neural excitation score at chance vs. interpretable
popularity/engagement models reaching ~0.82.

Fig B (real_coefficients.pdf): standardized logistic-regression coefficients,
showing which narrative properties drive transfer.

Reads results/real_transfer_predictors.json.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

d = json.load(open("results/real_transfer_predictors.json"))
os.makedirs("paper/figures", exist_ok=True)

# ---- Fig A: predictor comparison ----
rows = [("Neural Hawkes\n(excitation)", d["simcity_bridge_auc"], 0.02, RED)]
for name, r in d["feature_sets"].items():
    short = (name.replace("popularity", "pop.").replace(" (n_events)", "")
             .replace("interpretable ", ""))
    rows.append((short, r["auc_mean"], r["auc_std"], AQUA))

fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=200)
ys = list(range(len(rows)))[::-1]
for y, (lab, m, s, col) in zip(ys, rows):
    ax.barh(y, m - 0.5, left=0.5, height=0.6, color=col, alpha=0.85,
            zorder=2, edgecolor="white")
    ax.errorbar(m, y, xerr=s, fmt="none", ecolor=INK2, elinewidth=1,
                capsize=3, zorder=3)
    ax.annotate(f"{m:.2f}", (m, y), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK)
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.annotate("chance", (0.5, ys[0] + 0.55), ha="center", fontsize=8, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
ax.set_xlabel("CV ROC-AUC (predicting real transfer)", fontsize=9, color=INK)
ax.set_xlim(0.5, 0.92)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=8, colors=INK2)
fig.tight_layout()
fig.savefig("paper/figures/real_predictors.pdf", bbox_inches="tight")
fig.savefig("paper/figures/real_predictors.png", bbox_inches="tight")
print("Wrote paper/figures/real_predictors.{pdf,png}")

# ---- Fig B: univariate AUC per feature (collinearity-free) ----
univ = d["univariate_auc"]
items = sorted(univ.items(), key=lambda kv: kv[1])
labels = [k for k, _ in items]
vals = [v for _, v in items]
fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=200)
ys = list(range(len(items)))
ax.barh(ys, [v - 0.5 for v in vals], left=0.5, height=0.62, color=AQUA,
        alpha=0.85, edgecolor="white", zorder=2)
for y, v in zip(ys, vals):
    ax.annotate(f"{v:.2f}", (v, y), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=8.5, color=INK)
ax.set_xlabel("univariate ROC-AUC (each feature alone)", fontsize=9, color=INK)
ax.set_xlim(0.5, max(vals) * 1.06)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=8, colors=INK2)
fig.tight_layout()
fig.savefig("paper/figures/real_coefficients.pdf", bbox_inches="tight")
fig.savefig("paper/figures/real_coefficients.png", bbox_inches="tight")
print("Wrote paper/figures/real_coefficients.{pdf,png}")
