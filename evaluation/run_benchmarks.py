"""Aggregate SimCity benchmark numbers into a single results table.

This script collects the runnable benchmarks in the repo and emits both a JSON
record and a Markdown table under ``results/``. It is intentionally dependency
-light: the classical Hawkes baseline and the naive virality baseline run
without PyTorch Geometric; the full SimCity model numbers are read from
``results/simcity_metrics.json`` (produced by ``train.py``) when present.

Usage:
    python run_benchmarks.py --data data/synthetic_events.pkl --hawkes-epochs 8
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_hawkes_baseline import train_baseline


def naive_virality_baseline(data_path, train_fraction=0.70, val_fraction=0.10):
    """Predict the train-mean log-engagement for every test event."""
    df = pd.read_pickle(data_path).sort_values("t").reset_index(drop=True)
    y = df["log_engagement"].to_numpy(dtype=np.float64)
    t = df["t"].to_numpy(dtype=np.float64)

    t_min, t_max = t.min(), t.max()
    train_end = t_min + train_fraction * (t_max - t_min)
    val_end = train_end + val_fraction * (t_max - t_min)

    train_mask = t < train_end
    test_mask = t >= val_end

    train_mean = float(np.nanmean(y[train_mask]))
    y_test = y[test_mask]
    y_test = y_test[~np.isnan(y_test)]
    preds = np.full_like(y_test, train_mean)

    mae = float(np.abs(preds - y_test).mean())
    rmse = float(np.sqrt(((preds - y_test) ** 2).mean()))
    return {"virality_mae": mae, "virality_rmse": rmse, "train_mean": train_mean}


def to_markdown(record):
    lines = [
        "# SimCity Benchmark Results",
        "",
        f"Dataset: `{record['dataset']}`  ({record['n_events']} events, "
        "chronological 70/10/20 split)",
        "",
        "## Virality regression (log-engagement, held-out test)",
        "",
        "| Model | MAE ↓ | RMSE ↓ |",
        "|---|---|---|",
    ]
    v = record["virality"]
    lines.append(
        f"| Naive train-mean | {v['naive']['virality_mae']:.4f} | "
        f"{v['naive']['virality_rmse']:.4f} |"
    )
    for key, label in (("static_gnn_seir", "Static GNN + SEIR"),
                       ("vanilla_tgn", "Vanilla TGN")):
        b = v.get(key)
        if b:
            lines.append(
                f"| {label} | {b['virality_mae']:.4f} | {b['virality_rmse']:.4f} |"
            )
    if v.get("simcity"):
        s = v["simcity"]
        mae = s.get("virality_mae")
        rmse = s.get("virality_rmse")
        mae_s = f"{mae:.4f}" if mae is not None else "n/a"
        rmse_s = f"{rmse:.4f}" if rmse is not None else "n/a"
        lines.append(f"| **SimCity (full)** | {mae_s} | {rmse_s} |")
    else:
        lines.append("| **SimCity (full)** | _run train.py_ | _run train.py_ |")

    lines += [
        "",
        "## Cross-platform burst timing (Hawkes NLL, held-out test)",
        "",
        "| Model | NLL ↓ |",
        "|---|---|",
        f"| Static MHP baseline | {record['hawkes']['static_mhp_test_nll']:.4f} |",
    ]
    if v.get("simcity") and v["simcity"].get("hawkes_nll") is not None:
        lines.append(f"| **SimCity (full)** | {v['simcity']['hawkes_nll']:.4f} |")
    else:
        lines.append("| **SimCity (full)** | _run train.py_ |")
    lines.append("")
    lines.append(
        "> Note: numbers above are on **synthetic** multiplex data. Replace with "
        "the real ingested Reddit/HN/GDELT dataset before paper submission."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_events.pkl")
    parser.add_argument("--hawkes-epochs", type=int, default=8)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_pickle(args.data)
    n_events = int(len(df))

    print("=== Naive virality baseline ===")
    naive = naive_virality_baseline(args.data)
    print(naive)

    print("\n=== Static MHP (Hawkes NLL) baseline ===")
    hawkes = train_baseline(args.data, epochs=args.hawkes_epochs)

    simcity = None
    simcity_path = Path(args.out_dir) / "simcity_metrics.json"
    if simcity_path.exists():
        simcity = json.loads(simcity_path.read_text(encoding="utf-8")).get("test")

    graph_baselines = {}
    gb_path = Path(args.out_dir) / "graph_baselines.json"
    if gb_path.exists():
        graph_baselines = json.loads(gb_path.read_text(encoding="utf-8"))

    record = {
        "dataset": args.data,
        "n_events": n_events,
        "virality": {
            "naive": naive,
            "static_gnn_seir": graph_baselines.get("static_gnn_seir"),
            "vanilla_tgn": graph_baselines.get("vanilla_tgn"),
            "simcity": simcity,
        },
        "hawkes": {"static_mhp_test_nll": hawkes["test_nll"]},
    }

    json_path = Path(args.out_dir) / "benchmark_table.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    md_path = Path(args.out_dir) / "benchmark_table.md"
    md_path.write_text(to_markdown(record), encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
