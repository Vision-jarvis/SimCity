"""Generate the paper's main results figure (figures/transfer_auc.pdf).

Dot + interval plot of narrative transfer-detection AUC. Synthetic benchmark
(filled blue, 3 seeds, bootstrap 95% CI) vs. live HN+GDELT corpus (open green
circles, one per training seed -- the real-data score is NOT seed-stable, and
the figure shows that honestly rather than averaging it away). A dot plot is
used (not bars) because AUC is referenced to 0.5, not 0.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, AQUA = "#2a78d6", "#1baf7a"          # validated categorical pair
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# (label, synthetic value, syn CI, real per-seed values)
ROWS = [
    ("SimCity bridge score",          0.653, (0.580, 0.730), [0.319, 0.632, 0.248]),
    ("Bridge, count-stratified",      0.660, (0.656, 0.662), [0.167, 0.580, 0.199]),
    ("Popularity baseline",           0.551, None,           [0.612]),
    ("Static MHP (by construction)",  0.500, None,           [0.500]),
]
OFF = 0.16  # vertical offset between the two series within a row

fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ys = list(range(len(ROWS)))[::-1]  # top row first
for y, (label, sv, sci, rvals) in zip(ys, ROWS):
    # synthetic: filled dot + CI whisker
    if sci is not None:
        ax.plot(sci, [y + OFF, y + OFF], color=BLUE, lw=2,
                solid_capstyle="round", alpha=0.45, zorder=2)
    ax.plot(sv, y + OFF, "o", ms=7, color=BLUE, mec="white", mew=1.2, zorder=3)
    ax.annotate(f"{sv:.3f}", (sv, y + OFF), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=7.5, color=INK2)
    # real: one open circle per seed (no whisker -- seed-unstable)
    for rv in rvals:
        ax.plot(rv, y - OFF, "o", ms=6.5, mfc="white", mec=AQUA, mew=1.8, zorder=3)
    if len(rvals) == 1 and rvals[0] != 0.5:  # 0.5 already labeled on the row
        ax.annotate(f"{rvals[0]:.3f}", (rvals[0], y - OFF),
                    textcoords="offset points", xytext=(0, -13), ha="center",
                    fontsize=7.5, color=INK2)

ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.annotate("random (0.5)", (0.5, ys[0] + 0.52), ha="center", fontsize=7.5,
            color=INK2)

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=9, color=INK)
ax.set_xlabel("Transfer-detection AUC", fontsize=9, color=INK)
ax.set_xlim(0.13, 0.78)
ax.set_ylim(-0.6, len(ROWS) - 0.25)
ax.tick_params(axis="x", labelsize=8, colors=INK2)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)

handles = [
    plt.Line2D([], [], marker="o", ls="", ms=7, color=BLUE, mec="white",
               label="Synthetic (3 seeds; 95% CI)"),
    plt.Line2D([], [], marker="o", ls="", ms=6.5, mfc="white", mec=AQUA,
               mew=1.8, label="Real HN+GDELT (per seed; unstable)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)

os.makedirs("figures", exist_ok=True)
fig.tight_layout()
fig.savefig("figures/transfer_auc.pdf", bbox_inches="tight")
fig.savefig("figures/transfer_auc.png", bbox_inches="tight")
print("Wrote figures/transfer_auc.{pdf,png}")
