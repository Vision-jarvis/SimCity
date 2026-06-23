"""Tests for the counterfactual intervention simulator."""

import pytest

from simulation.intervention import InterventionSimulator, Intervention


BASE_PARAMS = {
    "N": 10000, "initial_S": 8000, "initial_E": 500, "initial_I": 0,
    "initial_R": 1500, "initial_Z": 0, "initial_D": 0,
    "theta": 2.0, "sigma": 0.5, "gamma_I": 0.1, "delta_D": 0.01,
    "base_beta_macro": 1.2, "baseline_lambda": 1.0, "injected_lambda": 50.0,
    "decay_gamma": 0.2, "phi": 0.05, "steps": 40, "dt": 1.0,
}


def test_unknown_intervention_rejected():
    with pytest.raises(ValueError):
        Intervention(type="not_a_real_thing")


def test_baseline_runs_and_has_history():
    sim = InterventionSimulator()
    history = sim.run_trajectory(BASE_PARAMS)
    assert len(history) == BASE_PARAMS["steps"]
    assert all({"S", "E", "I", "R", "Z", "D"} <= set(h) for h in history)


def test_determinism():
    """Same seed + same interventions -> identical trajectory."""
    sim = InterventionSimulator(seed=7)
    h1 = sim.run_trajectory(BASE_PARAMS)
    h2 = sim.run_trajectory(BASE_PARAMS)
    assert [h["I"] for h in h1] == [h["I"] for h in h2]


def test_fact_check_reduces_spread():
    """A fact-check should reduce peak infection vs. baseline."""
    sim = InterventionSimulator()
    report = sim.compare(
        BASE_PARAMS,
        [Intervention(type="fact_check", start_step=5, magnitude=0.7)],
        include_history=False,
    )
    assert report.treatment_metrics["peak_I"] <= report.baseline_metrics["peak_I"]
    assert report.deltas["total_reach"] <= 0  # fewer people reached


def test_influencer_amplify_increases_spread():
    """Adversarial amplification should increase reach vs. baseline."""
    sim = InterventionSimulator()
    report = sim.compare(
        BASE_PARAMS,
        [Intervention(type="influencer_amplify", start_step=0, magnitude=0.8)],
        include_history=False,
    )
    assert report.treatment_metrics["total_reach"] >= report.baseline_metrics["total_reach"]


def test_report_structure():
    sim = InterventionSimulator()
    report = sim.compare(
        BASE_PARAMS,
        [Intervention(type="deplatform_bots", start_step=3, magnitude=0.9)],
    )
    d = report.to_dict()
    for key in ("baseline_metrics", "treatment_metrics", "deltas", "pct_change", "interventions"):
        assert key in d
    assert d["interventions"][0]["type"] == "deplatform_bots"
    assert len(d["treatment_history"]) == BASE_PARAMS["steps"]
