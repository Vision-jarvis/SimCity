import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from analysis_tools.cascade_monitor import HawkesCascadeMonitor
from baselines.mhp import StaticHawkesBaseline
from run_hawkes_baseline import (
    chronological_splits,
    iter_windows,
    load_event_tensors,
    train_baseline,
)


def monitor_split(model, tensors, indices, batch_size, rolling_window, warning_z, critical_z):
    monitor = HawkesCascadeMonitor(
        num_platforms=model.num_platforms,
        rolling_window=rolling_window,
        warning_z=warning_z,
        critical_z=critical_z,
    )
    rows = []

    model.eval()
    model.reset_state()
    with torch.no_grad():
        for idx in iter_windows(indices, batch_size):
            mu, alpha, gamma = model.parameters_for_events(tensors["gdelt"][idx])
            scores = monitor.score(
                tensors["t"][idx],
                tensors["platform"][idx],
                mu,
                alpha,
                gamma,
                update_state=True,
            )
            for offset in range(scores["t"].numel()):
                rows.append(
                    {
                        "event_index": int(idx[scores["event_index"][offset]].item()),
                        "t": float(scores["t"][offset].item()),
                        "platform": int(scores["platform"][offset].item()),
                        "event_intensity": float(scores["event_intensity"][offset].item()),
                        "total_intensity": float(scores["total_intensity"][offset].item()),
                        "compensator": float(scores["compensator"][offset].item()),
                        "event_nll": float(scores["event_nll"][offset].item()),
                        "excitation_mass": float(scores["excitation_mass"][offset].item()),
                        "rolling_z": float(scores["rolling_z"][offset].item()),
                        "alert_level": int(scores["alert_level"][offset].item()),
                    }
                )

    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored, {}

    summary = {
        "events_scored": int(len(scored)),
        "warning_alerts": int((scored["alert_level"] == 1).sum()),
        "critical_alerts": int((scored["alert_level"] == 2).sum()),
        "mean_event_nll": float(scored["event_nll"].mean()),
        "p95_event_nll": float(scored["event_nll"].quantile(0.95)),
        "max_rolling_z": float(scored["rolling_z"].max()),
    }
    return scored, summary


def run_monitor(
    data_path,
    epochs=4,
    batch_size=512,
    lr=0.03,
    rolling_window=128,
    warning_z=2.0,
    critical_z=3.0,
    top_k=10,
):
    baseline = train_baseline(
        data_path=data_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    tensors = load_event_tensors(data_path)
    _, _, test_idx = chronological_splits(tensors["t"])
    model = StaticHawkesBaseline()
    with torch.no_grad():
        model.mu_logits.copy_(_inverse_softplus_tensor(torch.tensor(baseline["mu"])))
        model.gdelt_logits.copy_(
            _inverse_softplus_tensor(torch.tensor(baseline["gdelt_weight"]))
        )
        model.alpha_logits.copy_(_inverse_softplus_tensor(torch.tensor(baseline["alpha"])))
        model.gamma_logits.copy_(_inverse_softplus_tensor(torch.tensor(baseline["gamma"])))

    scored, summary = monitor_split(
        model,
        tensors,
        test_idx,
        batch_size=batch_size,
        rolling_window=rolling_window,
        warning_z=warning_z,
        critical_z=critical_z,
    )
    top_alerts = (
        scored.sort_values(["alert_level", "rolling_z", "event_nll"], ascending=False)
        .head(top_k)
        .to_dict(orient="records")
        if not scored.empty
        else []
    )
    report = {"baseline": baseline, "monitor_summary": summary, "top_alerts": top_alerts}
    print(json.dumps({"monitor_summary": summary, "top_alerts": top_alerts[:3]}, indent=2))
    return report, scored


def _inverse_softplus_tensor(value):
    return torch.log(torch.expm1(value).clamp_min(1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_events.pkl")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--rolling-window", type=int, default=128)
    parser.add_argument("--warning-z", type=float, default=2.0)
    parser.add_argument("--critical-z", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--events-out", default=None)
    args = parser.parse_args()

    report, scored = run_monitor(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rolling_window=args.rolling_window,
        warning_z=args.warning_z,
        critical_z=args.critical_z,
        top_k=args.top_k,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote monitor report to {out_path}")

    if args.events_out:
        events_path = Path(args.events_out)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(events_path, index=False)
        print(f"Wrote scored events to {events_path}")


if __name__ == "__main__":
    main()
