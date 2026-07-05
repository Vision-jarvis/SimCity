"""Within-narrative residual metrics — the evaluation that actually tests the
temporal/excitation signal instead of a narrative's static reach.

For each narrative we subtract that narrative's *mean* engagement (over its test
events) from both the truth and each model's prediction. The remaining residual
is purely the within-narrative, time-varying component driven by cross-platform
Hawkes excitation. A model that only knows a narrative's average reach (e.g. a
static snapshot GNN) has near-zero explanatory power on this residual; a model
that tracks temporal excitation state should do materially better.

Reports, per model:
  * raw MAE                      (reach-dominated; static models look good)
  * residual MAE                 (temporal signal; lower beats the mean predictor)
  * skill vs mean-predictor      (1 - residual_MAE / mean|residual_true|); >0 = captures temporal signal
  * within-narrative Spearman    (rank correlation of pred vs true surges)

Usage:
    python compute_residual_metrics.py
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr

MODELS = {
    "SimCity (full)": "results/simcity_test_preds.npz",
    "Vanilla TGN": "results/vanilla_tgn_test_preds.npz",
    "Static GNN + SEIR": "results/static_gnn_test_preds.npz",
}


def _demean_by_group(values, groups):
    """Subtract per-group mean from ``values``."""
    out = np.zeros_like(values, dtype=np.float64)
    for g in np.unique(groups):
        idx = groups == g
        out[idx] = values[idx] - values[idx].mean()
    return out


def evaluate(path):
    d = np.load(path)
    dst = d["dst"].astype(np.int64)
    true = d["true"].astype(np.float64)
    pred = d["pred"].astype(np.float64)

    raw_mae = float(np.abs(pred - true).mean())

    res_true = _demean_by_group(true, dst)
    res_pred = _demean_by_group(pred, dst)

    # Only narratives with >1 test event carry temporal signal.
    counts = {g: (dst == g).sum() for g in np.unique(dst)}
    multi = np.array([counts[g] > 1 for g in dst])
    rt, rp = res_true[multi], res_pred[multi]

    residual_mae = float(np.abs(rp - rt).mean())
    mean_predictor_mae = float(np.abs(rt).mean())  # error of predicting narrative mean
    skill = float(1.0 - residual_mae / mean_predictor_mae) if mean_predictor_mae > 0 else 0.0

    # Within-narrative Spearman, averaged over narratives with >=3 events.
    rhos = []
    for g in np.unique(dst):
        idx = dst == g
        if idx.sum() >= 3 and np.std(true[idx]) > 0 and np.std(pred[idx]) > 0:
            rho, _ = spearmanr(true[idx], pred[idx])
            if not np.isnan(rho):
                rhos.append(rho)
    mean_spearman = float(np.mean(rhos)) if rhos else float("nan")

    return {
        "raw_mae": raw_mae,
        "residual_mae": residual_mae,
        "mean_predictor_mae": mean_predictor_mae,
        "skill_vs_mean": skill,
        "within_narrative_spearman": mean_spearman,
        "n_test_events": int(len(true)),
    }


def main():
    results = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            results[name] = evaluate(path)
        else:
            print(f"[skip] {name}: {path} not found (run the model first)")

    os.makedirs("results", exist_ok=True)
    with open("results/residual_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown
    lines = [
        "# Temporal-signal evaluation (within-narrative residuals)",
        "",
        "Isolates the time-varying, excitation-driven component of engagement by "
        "removing each narrative's mean. `skill_vs_mean` > 0 means the model beats "
        "a narrative-mean predictor; Spearman measures within-narrative surge ranking.",
        "",
        "| Model | Raw MAE | Residual MAE ↓ | Skill vs mean ↑ | Within-narr. Spearman ↑ |",
        "|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['raw_mae']:.4f} | {m['residual_mae']:.4f} | "
            f"{m['skill_vs_mean']:+.4f} | {m['within_narrative_spearman']:.4f} |"
        )
    lines.append("")
    if results:
        mp = next(iter(results.values()))["mean_predictor_mae"]
        lines.append(f"> Narrative-mean predictor residual MAE = {mp:.4f} (skill = 0 baseline).")
    md = "\n".join(lines) + "\n"
    with open("results/residual_metrics.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote results/residual_metrics.md and results/residual_metrics.json")


if __name__ == "__main__":
    main()
