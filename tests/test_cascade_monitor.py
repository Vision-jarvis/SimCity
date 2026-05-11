import pandas as pd
import torch

from analysis_tools.cascade_monitor import HawkesCascadeMonitor
from baselines.mhp import StaticHawkesBaseline
from run_cascade_monitor import monitor_split, run_monitor
from run_hawkes_baseline import load_event_tensors


def test_cascade_monitor_scores_events_and_updates_state():
    monitor = HawkesCascadeMonitor(num_platforms=3, rolling_window=3, warning_z=1.0)
    t = torch.tensor([0.0, 60.0, 120.0, 7200.0])
    platform = torch.tensor([0, 1, 1, 2])
    mu = torch.full((4, 3), 0.3)
    alpha = torch.full((4, 3, 3), 0.2)
    gamma = torch.full((4, 3, 3), 1.0)

    scores = monitor.score(t, platform, mu, alpha, gamma)

    assert scores["event_nll"].shape == (4,)
    assert scores["event_intensity"].shape == (4,)
    assert torch.isfinite(scores["event_nll"]).all()
    assert monitor.excitation_state.sum() > 0
    assert torch.isfinite(monitor.last_time)


def test_cascade_monitor_reset_removes_history():
    monitor = HawkesCascadeMonitor(num_platforms=3)
    t = torch.tensor([0.0, 60.0])
    platform = torch.tensor([0, 1])
    mu = torch.full((2, 3), 0.3)
    alpha = torch.full((2, 3, 3), 0.2)
    gamma = torch.full((2, 3, 3), 1.0)

    monitor.score(t, platform, mu, alpha, gamma)
    monitor.reset_state()

    assert monitor.excitation_state.sum() == 0
    assert torch.isnan(monitor.last_time)


def test_monitor_split_returns_summary_and_alert_columns(tmp_path):
    df = pd.DataFrame(
        {
            "t": [0.0, 60.0, 120.0, 240.0, 7200.0, 7260.0],
            "platform": [0, 1, 1, 2, 0, 1],
            "gdelt_volume": [0.0, 1.0, 4.0, 2.0, 15.0, 7.0],
        }
    )
    path = tmp_path / "events.pkl"
    df.to_pickle(path)
    tensors = load_event_tensors(path)
    model = StaticHawkesBaseline(num_platforms=3)
    indices = torch.arange(len(df))

    scored, summary = monitor_split(
        model,
        tensors,
        indices,
        batch_size=3,
        rolling_window=3,
        warning_z=1.0,
        critical_z=2.0,
    )

    assert len(scored) == len(df)
    assert summary["events_scored"] == len(df)
    assert {"event_nll", "rolling_z", "alert_level"}.issubset(scored.columns)


def test_run_monitor_end_to_end_on_tiny_pickle(tmp_path):
    df = pd.DataFrame(
        {
            "t": [0.0, 60.0, 120.0, 240.0, 420.0, 900.0, 1200.0, 1800.0],
            "platform": [0, 1, 0, 1, 2, 0, 1, 2],
            "gdelt_volume": [0.0, 3.0, 8.0, 2.0, 1.0, 5.0, 9.0, 4.0],
        }
    )
    path = tmp_path / "events.pkl"
    df.to_pickle(path)

    report, scored = run_monitor(path, epochs=1, batch_size=3, top_k=2)

    assert report["monitor_summary"]["events_scored"] > 0
    assert len(report["top_alerts"]) <= 2
    assert not scored.empty
