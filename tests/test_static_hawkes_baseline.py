import pandas as pd
import torch

from baselines.mhp import StaticHawkesBaseline
from run_hawkes_baseline import chronological_splits, load_event_tensors, train_baseline


def test_static_hawkes_baseline_parameters_are_positive_and_differentiable():
    model = StaticHawkesBaseline(num_platforms=3)
    t = torch.tensor([0.0, 60.0, 180.0, 420.0])
    platform = torch.tensor([0, 1, 1, 2])
    gdelt = torch.tensor([0.0, 5.0, 10.0, 2.0])

    loss = model(t, platform, gdelt)
    loss.backward()
    params = model.learned_parameters()

    assert torch.isfinite(loss)
    assert model.alpha_logits.grad is not None
    assert model.gamma_logits.grad is not None
    assert (params["mu"] > 0).all()
    assert (params["gdelt_weight"] > 0).all()
    assert (params["alpha"] > 0).all()
    assert (params["gamma"] > 0).all()


def test_chronological_splits_preserve_order_and_non_overlap():
    t = torch.arange(10, dtype=torch.float)
    train_idx, val_idx, test_idx = chronological_splits(t, train_fraction=0.6, val_fraction=0.2)

    assert train_idx.max() < val_idx.min()
    assert val_idx.max() < test_idx.min()
    assert train_idx.numel() + val_idx.numel() + test_idx.numel() == t.numel()


def test_train_baseline_runs_on_tiny_pickle(tmp_path):
    df = pd.DataFrame(
        {
            "t": [0.0, 60.0, 120.0, 240.0, 420.0, 900.0, 1200.0, 1800.0],
            "platform": [0, 1, 0, 1, 2, 0, 1, 2],
            "gdelt_volume": [0.0, 3.0, 8.0, 2.0, 1.0, 5.0, 9.0, 4.0],
        }
    )
    path = tmp_path / "events.pkl"
    df.to_pickle(path)

    tensors = load_event_tensors(path)
    assert tensors["t"].shape[0] == len(df)

    result = train_baseline(path, epochs=1, batch_size=3, lr=0.01)

    assert result["train_events"] > 0
    assert result["val_events"] > 0
    assert result["test_events"] > 0
    assert torch.isfinite(torch.tensor(result["test_nll"]))
    assert len(result["history"]) == 1
    assert len(result["alpha"]) == 3
