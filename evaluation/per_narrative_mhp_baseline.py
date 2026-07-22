"""Step 1 of the revision plan: the baseline that decides everything.

Reviewer critique: the static MHP baseline is AUC 0.5 *by construction* (one
global alpha cannot rank narratives), so beating it proves little. The proper
competitor is a **per-narrative** Hawkes process --- separate parameters fitted
independently for each narrative, with no graph/TGN conditioning at all. If the
TGN-conditioned bridge score cannot beat that, the conditioning adds nothing.

Structural fact this script exposes: by construction a narrative's *pre-switch*
window (the only data a causal transfer predictor may use) contains events on a
single platform. A per-narrative fit therefore cannot identify cross-platform
alpha_{ij}, i != j --- there are no cross-platform events to fit. What it *can*
identify is the narrative's own excitation dynamics. This is exactly the
amortisation gap the neural head is supposed to close: it learns a shared map
from narrative state to excitation parameters across all narratives, whereas
independent per-narrative MLE has only that narrative's handful of events
(median 3).

To be maximally generous to the baseline we fit a univariate exponential-kernel
Hawkes process per narrative by exact MLE and score narratives by *every*
plausible statistic derived from it (branching ratio, alpha, mu, plus
model-free burstiness measures), then report the **best** AUC achieved by any
of them. SimCity must beat that best-case baseline.

Usage:
    python -m evaluation.per_narrative_mhp_baseline \
        --preds results/simcity_test_preds_real_s2.npz \
        --orient-with results/simcity_val_preds_real_s2.npz
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from evaluation.narrative_transfer_eval import build_table, stratified_auc

TIME_SCALE = 3600.0  # seconds -> hours, for numerical conditioning


def hawkes_nll(params, t):
    """Exact NLL of a univariate exponential-kernel Hawkes process.

    lambda(t) = mu + sum_{t_j < t} alpha * exp(-gamma (t - t_j))
    Uses the standard O(n) recursion for the excitation state.
    """
    log_mu, log_alpha, log_gamma = params
    mu, alpha, gamma = np.exp(log_mu), np.exp(log_alpha), np.exp(log_gamma)

    T = t[-1] - t[0]
    tt = t - t[0]

    # Recursive excitation: A_i = sum_{j<i} exp(-gamma (t_i - t_j))
    A = 0.0
    loglik = 0.0
    for i in range(len(tt)):
        if i > 0:
            A = np.exp(-gamma * (tt[i] - tt[i - 1])) * (1.0 + A)
        lam = mu + alpha * A
        loglik += np.log(max(lam, 1e-12))

    # Compensator: mu*T + (alpha/gamma) * sum_j (1 - exp(-gamma (T - t_j)))
    comp = mu * T + (alpha / gamma) * np.sum(1.0 - np.exp(-gamma * (T - tt)))
    return -(loglik - comp)


def fit_narrative(t, n_restarts=3, seed=0):
    """MLE fit; returns dict of fitted params (nan if unfittable)."""
    if len(t) < 3 or (t[-1] - t[0]) <= 0:
        return None
    rng = np.random.default_rng(seed)
    best, best_nll = None, np.inf
    duration = (t[-1] - t[0]) / TIME_SCALE
    rate0 = len(t) / max(duration, 1e-6)
    inits = [np.log([max(rate0, 1e-3), 0.5, 1.0]),
             np.log([max(rate0 * 0.5, 1e-3), 1.0, 2.0]),
             np.log([max(rate0 * 0.2, 1e-3), 0.2, 0.5])]
    ts = t / TIME_SCALE
    for k in range(min(n_restarts, len(inits))):
        x0 = inits[k] + rng.normal(0, 0.1, 3)
        try:
            res = minimize(hawkes_nll, x0, args=(ts,), method="Nelder-Mead",
                           options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4})
            if res.fun < best_nll:
                best_nll, best = res.fun, res.x
        except Exception:
            continue
    if best is None:
        return None
    mu, alpha, gamma = np.exp(best)
    return {"mu": mu, "alpha": alpha, "gamma": gamma,
            "branching": alpha / max(gamma, 1e-9), "nll": best_nll}


def model_free_stats(t):
    """Burstiness statistics that need no model fit."""
    n = len(t)
    dur = (t[-1] - t[0]) / TIME_SCALE if n > 1 else 0.0
    out = {"n_events": float(n), "rate": n / dur if dur > 0 else 0.0,
           "duration": dur}
    if n >= 3:
        dt = np.diff(t) / TIME_SCALE
        m = dt.mean()
        out["ia_cv"] = float(dt.std() / m) if m > 0 else 0.0
        # Burstiness coefficient (Goh & Barabasi)
        s, mm = dt.std(), dt.mean()
        out["burstiness"] = float((s - mm) / (s + mm)) if (s + mm) > 0 else 0.0
    else:
        out["ia_cv"] = 0.0
        out["burstiness"] = 0.0
    return out


def narrative_features(npz_path):
    """Per-narrative baseline statistics + labels for one split dump."""
    d = np.load(npz_path)
    dst = d["dst"].astype(np.int64)
    plat = d["event_platform"].astype(np.int64)
    tt = d["event_t"].astype(float)
    rows = build_table(d)

    feats = {k: [] for k in ["branching", "alpha", "mu", "gamma",
                             "n_events", "rate", "ia_cv", "burstiness"]}
    n_fit = 0
    for r in rows:
        g = r["narrative"]
        idx = np.where(dst == g)[0]
        order = idx[np.argsort(tt[idx])]
        p = plat[order]
        sw = np.where(p != p[0])[0]
        pre = order[: sw[0]] if sw.size else order
        t_pre = np.sort(tt[pre])
        mf = model_free_stats(t_pre)
        fit = fit_narrative(t_pre)
        if fit is not None:
            n_fit += 1
        for k in ["branching", "alpha", "mu", "gamma"]:
            feats[k].append(fit[k] if fit else np.nan)
        for k in ["n_events", "rate", "ia_cv", "burstiness"]:
            feats[k].append(mf[k])

    clean = {}
    for k, v in feats.items():
        v = np.array(v, dtype=float)
        if np.all(np.isnan(v)):
            continue
        clean[k] = np.where(np.isnan(v), np.nanmedian(v), v)
    labels = np.array([r["label"] for r in rows])
    counts = np.array([r["n_pre_events"] for r in rows], dtype=float)
    bridge = np.array([r["bridge_score"] for r in rows])
    return clean, labels, counts, bridge, n_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/simcity_test_preds_real_s2.npz")
    ap.add_argument("--orient-with", default=None)
    ap.add_argument("--out", default="results/per_narrative_mhp_baseline.md")
    args = ap.parse_args()

    d = np.load(args.preds)
    for k in ("event_platform", "event_t"):
        if k not in d:
            raise SystemExit(f"{args.preds} lacks {k}; rerun train.py")

    dst = d["dst"].astype(np.int64)
    plat = d["event_platform"].astype(np.int64)
    tt = d["event_t"].astype(float)

    rows = build_table(d)
    labels = np.array([r["label"] for r in rows])
    counts = np.array([r["n_pre_events"] for r in rows], dtype=float)
    narrs = [r["narrative"] for r in rows]

    # SimCity bridge score (with the same validation orientation, if given)
    bridge = np.array([r["bridge_score"] for r in rows])
    orientation = 1.0
    if args.orient_with:
        vrows = build_table(np.load(args.orient_with))
        vlab = np.array([r["label"] for r in vrows])
        vsc = np.array([r["bridge_score"] for r in vrows])
        if 0 < vlab.sum() < len(vlab):
            orientation = 1.0 if roc_auc_score(vlab, vsc) >= 0.5 else -1.0
    bridge = orientation * bridge

    # Per-narrative Hawkes MLE on the pre-switch window
    feats = {k: [] for k in ["branching", "alpha", "mu", "gamma",
                             "n_events", "rate", "ia_cv", "burstiness"]}
    n_fit = 0
    for g in narrs:
        idx = np.where(dst == g)[0]
        order = idx[np.argsort(tt[idx])]
        p = plat[order]
        sw = np.where(p != p[0])[0]
        pre = order[: sw[0]] if sw.size else order
        t_pre = np.sort(tt[pre])

        mf = model_free_stats(t_pre)
        fit = fit_narrative(t_pre)
        if fit is not None:
            n_fit += 1
        for k in ["branching", "alpha", "mu", "gamma"]:
            feats[k].append(fit[k] if fit else np.nan)
        for k in ["n_events", "rate", "ia_cv", "burstiness"]:
            feats[k].append(mf[k])

    print(f"per-narrative Hawkes MLE: fitted {n_fit}/{len(narrs)} "
          f"(rest have <3 pre-switch events)")

    # Score narratives by every candidate statistic; nan -> median (neutral)
    results = {}
    for name, vals in feats.items():
        v = np.array(vals, dtype=float)
        if np.all(np.isnan(v)):
            continue
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        if np.std(v) == 0:
            continue
        auc = roc_auc_score(labels, v)
        # a baseline statistic may be anti-predictive; give it the better sign
        auc_oriented = max(auc, 1 - auc)
        results[name] = {"auc_raw": float(auc), "auc_best_sign": float(auc_oriented),
                         "strata": float(stratified_auc(labels, v if auc >= 0.5 else -v, counts))}

    sc_auc = float(roc_auc_score(labels, bridge))
    sc_strata = float(stratified_auc(labels, bridge, counts))
    best_name = max(results, key=lambda k: results[k]["auc_best_sign"])
    best = results[best_name]

    record = {
        "preds": args.preds,
        "n_narratives": int(len(labels)), "n_transfer": int(labels.sum()),
        "n_fitted": n_fit,
        "simcity_bridge": {"auc": sc_auc, "strata": sc_strata,
                           "orientation": orientation},
        "per_narrative_baseline": results,
        "best_baseline": {"statistic": best_name, **best},
        "verdict": ("SimCity beats best per-narrative statistic"
                    if sc_auc > best["auc_best_sign"] else
                    "per-narrative baseline matches or beats SimCity"),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/per_narrative_mhp_baseline.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    lines = [
        "# Per-narrative MHP baseline (revision Step 1)",
        "",
        f"Test narratives: {len(labels)} ({int(labels.sum())} transfer). "
        f"Per-narrative Hawkes MLE converged for {n_fit}.",
        "",
        "**Structural note:** a narrative's pre-switch window contains events on "
        "a *single* platform by construction, so per-narrative fitting cannot "
        "identify cross-platform alpha at all. It identifies the narrative's own "
        "excitation dynamics. Scores below are the most generous reading of that "
        "baseline (best sign taken for each statistic).",
        "",
        "| Scorer | AUC (best sign) | count-stratified |",
        "|---|---|---|",
    ]
    for k in sorted(results, key=lambda k: -results[k]["auc_best_sign"]):
        lines.append(f"| per-narrative: {k} | {results[k]['auc_best_sign']:.3f} "
                     f"| {results[k]['strata']:.3f} |")
    lines.append(f"| **SimCity bridge score** | **{sc_auc:.3f}** | {sc_strata:.3f} |")
    lines += ["", f"> Best per-narrative statistic: **{best_name}** "
                  f"(AUC {best['auc_best_sign']:.3f}). Verdict: {record['verdict']}.", ""]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines[-14:]))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
