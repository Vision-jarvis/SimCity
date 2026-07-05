"""Cross-platform next-event prediction — the timing/transfer task where a
Hawkes intensity model should win.

At each event, a model's conditional intensity vector lambda_i(t_k) (computed
from history strictly before t_k) predicts *which platform* fires. We score:
  - top-1 accuracy  (argmax lambda == actual platform)
  - macro one-vs-rest ROC-AUC (using normalized intensities as class scores)
  - log-loss

Compared models:
  - SimCity (neural, per-event mu/alpha/gamma)  -> results/simcity_test_preds_hawkes.npz
  - Static MHP (single global mu/alpha/gamma)   -> trained inline
  - Marginal frequency baseline                 -> train-set platform priors
  - Persistence baseline                        -> predict previous event's platform

This directly tests the paper's central claim: does per-narrative, graph-informed
neural excitation predict cross-platform transfer better than a static Hawkes?
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from baselines.mhp import StaticHawkesBaseline
from run_hawkes_baseline import load_event_tensors, chronological_splits, iter_windows

P = 3


def _metrics(intensity, platform):
    """intensity: (N,P) >=0 ; platform: (N,) true class."""
    prob = intensity / np.clip(intensity.sum(1, keepdims=True), 1e-9, None)
    pred = prob.argmax(1)
    acc = float(accuracy_score(platform, pred))
    present = sorted(set(int(x) for x in platform))
    try:
        if len(present) == 2:
            # binary AUC on the minority (rarer) class — robust to imbalance
            counts = {c: int((platform == c).sum()) for c in present}
            pos = min(counts, key=counts.get)
            score = prob[:, pos]
            auc = float(roc_auc_score((platform == pos).astype(int), score))
        elif len(present) > 2:
            auc = float(roc_auc_score(platform, prob, multi_class="ovr", average="macro"))
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")
    try:
        ll = float(log_loss(platform, prob, labels=list(range(P))))
    except ValueError:
        ll = float("nan")
    return {"top1_acc": acc, "macro_auc": auc, "log_loss": ll, "n": int(len(platform))}


def train_static_mhp_intensities(data_path, epochs=8):
    tensors = load_event_tensors(data_path, num_platforms=P)
    tr, va, te = chronological_splits(tensors["t"])
    model = StaticHawkesBaseline(num_platforms=P)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(epochs):
        model.train(); model.reset_state()
        for idx in iter_windows(tr, 512):
            opt.zero_grad()
            loss = model(tensors["t"][idx], tensors["platform"][idx], tensors["gdelt"][idx], update_state=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); model.detach_state()
    # Causal test intensities: reset at the test boundary, stream test events.
    model.eval(); model.reset_state()
    with torch.no_grad():
        mu, alpha, gamma = model.parameters_for_events(tensors["gdelt"][te])
        inten, plat = model.hawkes_loss.event_intensities(
            tensors["t"][te], tensors["platform"][te], mu, alpha, gamma, update_state=True
        )
    return inten.cpu().numpy(), plat.cpu().numpy(), (tr, va, te), tensors


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_events.pkl")
    args = ap.parse_args()
    data = args.data
    rows = {}

    # Static MHP
    mhp_int, mhp_plat, (tr, va, te), tensors = train_static_mhp_intensities(data)
    rows["Static MHP"] = _metrics(mhp_int, mhp_plat)

    # Baselines using the same test platforms
    plat_all = tensors["platform"].cpu().numpy()
    train_prior = np.bincount(plat_all[tr.cpu().numpy()], minlength=P).astype(float)
    train_prior /= train_prior.sum()
    N = len(mhp_plat)
    rows["Marginal prior"] = _metrics(np.tile(train_prior, (N, 1)), mhp_plat)

    # Persistence: predict previous event's platform (one-hot, chronological test order)
    te_plat_time_order = tensors["platform"][te].cpu().numpy()
    prev = np.concatenate([te_plat_time_order[:1], te_plat_time_order[:-1]])
    persist = np.eye(P)[prev] * 0.9 + 0.05  # smoothed one-hot
    rows["Persistence"] = _metrics(persist, te_plat_time_order)

    # SimCity (from dumped intensities)
    sc_path = "results/simcity_test_preds_hawkes.npz"
    if os.path.exists(sc_path):
        d = np.load(sc_path)
        rows["SimCity (neural Hawkes)"] = _metrics(
            d["intensity"].astype(float), d["platform"].astype(int)
        )
    else:
        print(f"[warn] {sc_path} missing — run train.py to dump neural intensities.")

    os.makedirs("results", exist_ok=True)
    with open("results/platform_prediction.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # --- Transition-only evaluation: isolate genuine cross-platform TRANSFER ---
    # Restrict to test events whose platform differs from the previous event, so a
    # trivial "predict previous platform" baseline is wrong by construction and the
    # metric measures actual transfer prediction rather than platform autocorrelation.
    trans = {}
    plat_seq = mhp_plat  # chronological test platforms
    tmask = np.concatenate([[False], plat_seq[1:] != plat_seq[:-1]])
    if tmask.sum() >= 10:
        trans["Static MHP"] = _metrics(mhp_int[tmask], mhp_plat[tmask])
        if os.path.exists(sc_path):
            d = np.load(sc_path)
            si, sp = d["intensity"].astype(float), d["platform"].astype(int)
            if len(sp) == len(tmask):
                trans["SimCity (neural Hawkes)"] = _metrics(si[tmask], sp[tmask])
        trans["Marginal prior"] = _metrics(np.tile(train_prior, (int(tmask.sum()), 1)), mhp_plat[tmask])

    order = ["SimCity (neural Hawkes)", "Static MHP", "Persistence", "Marginal prior"]
    lines = [
        "# Cross-platform next-event prediction (timing / transfer)",
        "",
        "Which platform fires next, from the conditional intensity lambda_i(t_k). "
        f"Random top-1 acc = {1/P:.3f}; random AUC = 0.5.",
        "",
        "## All test events",
        "| Model | Top-1 acc ^ | AUC ^ | Log-loss v |",
        "|---|---|---|---|",
    ]
    for name in order:
        if name in rows:
            m = rows[name]
            lines.append(f"| {name} | {m['top1_acc']:.3f} | {m['macro_auc']:.3f} | {m['log_loss']:.3f} |")
    lines += [
        "",
        "## Transition events only (platform changes -> real transfer; persistence = 0 by construction)",
        f"_{int(tmask.sum())} transition events of {len(plat_seq)} test events._",
        "",
        "| Model | Top-1 acc ^ | AUC ^ | Log-loss v |",
        "|---|---|---|---|",
    ]
    for name in ["SimCity (neural Hawkes)", "Static MHP", "Marginal prior"]:
        if name in trans:
            m = trans[name]
            lines.append(f"| {name} | {m['top1_acc']:.3f} | {m['macro_auc']:.3f} | {m['log_loss']:.3f} |")
    lines.append("")
    md = "\n".join(lines) + "\n"
    with open("results/platform_prediction.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote results/platform_prediction.md and results/platform_prediction.json")


if __name__ == "__main__":
    main()
