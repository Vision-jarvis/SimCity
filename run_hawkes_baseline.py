import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from baselines.mhp import StaticHawkesBaseline


def load_event_tensors(path, num_platforms=3, device="cpu"):
    try:
        df = pd.read_pickle(path).sort_values("t").reset_index(drop=True)
    except ModuleNotFoundError as exc:
        if "numpy._core" in str(exc):
            raise RuntimeError(
                "Could not read the pickle because it was created with a "
                "different NumPy serialization layout. Regenerate it in this "
                "environment with: python data/synthetic_generator.py "
                "--events 10000 --out data/synthetic_events.pkl"
            ) from exc
        raise
    required = {"t", "platform", "gdelt_volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    platform = torch.tensor(df["platform"].values, dtype=torch.long, device=device)
    if platform.min() < 0 or platform.max() >= num_platforms:
        raise ValueError(
            f"platform ids must be in [0, {num_platforms - 1}], "
            f"got min={platform.min().item()} max={platform.max().item()}"
        )

    return {
        "t": torch.tensor(df["t"].values, dtype=torch.float, device=device),
        "platform": platform,
        "gdelt": torch.tensor(df["gdelt_volume"].values, dtype=torch.float, device=device),
    }


def chronological_splits(t, train_fraction=0.70, val_fraction=0.10):
    t_min, t_max = float(t.min()), float(t.max())
    train_end = t_min + train_fraction * (t_max - t_min)
    val_end = train_end + val_fraction * (t_max - t_min)

    train_idx = torch.where(t < train_end)[0]
    val_idx = torch.where((t >= train_end) & (t < val_end))[0]
    test_idx = torch.where(t >= val_end)[0]
    if min(train_idx.numel(), val_idx.numel(), test_idx.numel()) == 0 and t.numel() >= 3:
        n = t.numel()
        train_end_idx = max(1, min(n - 2, int(n * train_fraction)))
        val_end_idx = max(train_end_idx + 1, min(n - 1, int(n * (train_fraction + val_fraction))))
        all_idx = torch.arange(n, device=t.device)
        train_idx = all_idx[:train_end_idx]
        val_idx = all_idx[train_end_idx:val_end_idx]
        test_idx = all_idx[val_end_idx:]
    return train_idx, val_idx, test_idx


def iter_windows(indices, batch_size):
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def evaluate_split(model, tensors, indices, batch_size):
    model.eval()
    model.reset_state()
    total_loss = 0.0
    total_events = 0

    with torch.no_grad():
        for idx in iter_windows(indices, batch_size):
            loss = model(
                tensors["t"][idx],
                tensors["platform"][idx],
                tensors["gdelt"][idx],
                update_state=True,
            )
            total_loss += loss.item() * idx.numel()
            total_events += idx.numel()

    return total_loss / max(total_events, 1)


def train_baseline(
    data_path,
    num_platforms=3,
    epochs=8,
    batch_size=512,
    lr=0.03,
    device="cpu",
):
    tensors = load_event_tensors(data_path, num_platforms=num_platforms, device=device)
    train_idx, val_idx, test_idx = chronological_splits(tensors["t"])

    model = StaticHawkesBaseline(num_platforms=num_platforms).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        model.reset_state()
        total_loss = 0.0
        total_events = 0

        for idx in iter_windows(train_idx, batch_size):
            optimizer.zero_grad()
            loss = model(
                tensors["t"][idx],
                tensors["platform"][idx],
                tensors["gdelt"][idx],
                update_state=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            model.detach_state()

            total_loss += loss.item() * idx.numel()
            total_events += idx.numel()

        train_nll = total_loss / max(total_events, 1)
        val_nll = evaluate_split(model, tensors, val_idx, batch_size)
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})
        print(
            f"Epoch {epoch:02d} | train Hawkes NLL: {train_nll:.4f} "
            f"| val Hawkes NLL: {val_nll:.4f}"
        )

    test_nll = evaluate_split(model, tensors, test_idx, batch_size)
    params = model.learned_parameters()
    result = {
        "train_events": int(train_idx.numel()),
        "val_events": int(val_idx.numel()),
        "test_events": int(test_idx.numel()),
        "history": history,
        "test_nll": test_nll,
        "mu": params["mu"].tolist(),
        "gdelt_weight": params["gdelt_weight"].tolist(),
        "alpha": params["alpha"].tolist(),
        "gamma": params["gamma"].tolist(),
    }
    print(f"Final test Hawkes NLL: {test_nll:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_events.pkl")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--num-platforms", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    result = train_baseline(
        data_path=args.data,
        num_platforms=args.num_platforms,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote baseline report to {out_path}")


if __name__ == "__main__":
    main()
