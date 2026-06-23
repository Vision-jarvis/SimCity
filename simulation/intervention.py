"""
Counterfactual intervention simulator — the digital-twin "what-if" engine.

Given a scenario, this runs a **baseline** trajectory and one or more
**counterfactual** trajectories in which interventions are applied at a chosen
step, then reports the measured deltas (peak misinformation, total reach,
persistent zealots, polarization, etc.).

This is what turns SimCity from a forward simulator into a *twin*: it answers
"what would happen if we deployed a fact-check at hour 10?" or "what if we
deplatformed the bot network?" by comparing worlds that differ only in the
intervention.

Interventions are expressed as multiplicative/additive modifiers on the
macroscopic SEIR-Z-D drivers:

| Type                | Effect on dynamics                                            |
|---------------------|---------------------------------------------------------------|
| `fact_check`        | ↓ beta (transmission), ↓ theta (algorithmic boost of Z)       |
| `counter_narrative` | ↓ beta, pushes content opinion toward neutral                 |
| `deplatform_bots`   | ↓ Hawkes lambda surge (→ ↓ zeta resurgence), ↓ theta          |
| `rate_limit`        | caps lambda at a ceiling (slows cascade growth)               |
| `influencer_amplify`| ↑ beta and re-injects a lambda spike (adversarial what-if)    |
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from simulation.seir_z_d import StochasticSEIRZD

logger = logging.getLogger(__name__)

INTERVENTION_TYPES = {
    "fact_check",
    "counter_narrative",
    "deplatform_bots",
    "rate_limit",
    "influencer_amplify",
}


@dataclass
class Intervention:
    """A single intervention applied from ``start_step`` onward."""

    type: str
    start_step: int = 0
    magnitude: float = 0.5  # strength in [0, 1] (or amplification factor for amplify)
    name: str = ""

    def __post_init__(self):
        if self.type not in INTERVENTION_TYPES:
            raise ValueError(
                f"Unknown intervention '{self.type}'. Valid: {sorted(INTERVENTION_TYPES)}"
            )
        if not self.name:
            self.name = self.type


@dataclass
class TrajectoryMetrics:
    """Summary metrics extracted from a simulation history."""

    peak_I: float = 0.0
    peak_I_step: int = 0
    total_reach: float = 0.0          # N - final_S: everyone ever exposed
    final_Z: float = 0.0              # persistent misinformation (zealots)
    final_D: float = 0.0             # archived / debunked
    auc_I: float = 0.0               # cumulative infected-time (engagement volume)
    final_mean_opinion: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "peak_I": round(self.peak_I, 2),
            "peak_I_step": self.peak_I_step,
            "total_reach": round(self.total_reach, 2),
            "final_Z": round(self.final_Z, 2),
            "final_D": round(self.final_D, 2),
            "auc_I": round(self.auc_I, 2),
            "final_mean_opinion": round(self.final_mean_opinion, 4),
        }


@dataclass
class InterventionReport:
    """Baseline vs. counterfactual comparison."""

    baseline_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    deltas: Dict[str, float]
    pct_change: Dict[str, float]
    interventions: List[Dict]
    baseline_history: List[Dict] = field(default_factory=list)
    treatment_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "baseline_metrics": self.baseline_metrics,
            "treatment_metrics": self.treatment_metrics,
            "deltas": self.deltas,
            "pct_change": self.pct_change,
            "interventions": self.interventions,
            "baseline_history": self.baseline_history,
            "treatment_history": self.treatment_history,
        }


class InterventionSimulator:
    """Runs baseline + counterfactual SEIR-Z-D trajectories and compares them."""

    def __init__(self, device: str = "cpu", seed: Optional[int] = 42):
        self.device = device
        self.seed = seed

    # ------------------------------------------------------------------ #
    def run_trajectory(
        self,
        params: Dict,
        interventions: Optional[List[Intervention]] = None,
    ) -> List[Dict]:
        """Step the SEIR-Z-D model forward, applying interventions per step."""
        interventions = interventions or []
        if self.seed is not None:
            # Deterministic so baseline/treatment differ ONLY by the intervention.
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        seir = StochasticSEIRZD(
            initial_state=[
                params["initial_S"], params["initial_E"], params["initial_I"],
                params["initial_R"], params["initial_Z"], params["initial_D"],
            ],
            N=params["N"],
            theta=params["theta"],
            sigma=params["sigma"],
            gamma_I=params["gamma_I"],
            delta_D=params["delta_D"],
            device=self.device,
        )

        base_theta = seir.theta
        base_beta = params["base_beta_macro"]
        current_lambda = params["injected_lambda"]
        baseline_lambda = params["baseline_lambda"]
        decay_gamma = params["decay_gamma"]
        phi = params["phi"]
        dt = params["dt"]
        steps = params["steps"]

        history: List[Dict] = []
        for step in range(steps):
            # Decay the Hawkes intensity analytically.
            current_lambda = current_lambda * np.exp(-decay_gamma * dt)

            # --- Apply active interventions to this step's effective params ---
            beta_eff = base_beta
            lambda_eff = current_lambda
            seir.theta = base_theta
            for iv in interventions:
                if step < iv.start_step:
                    continue
                m = iv.magnitude
                if iv.type in ("fact_check", "counter_narrative"):
                    beta_eff *= max(0.0, 1.0 - m)
                    seir.theta *= max(0.0, 1.0 - 0.5 * m)
                elif iv.type == "deplatform_bots":
                    lambda_eff *= max(0.0, 1.0 - m)
                    current_lambda *= max(0.0, 1.0 - m)  # persists into future decay
                    seir.theta *= max(0.0, 1.0 - 0.5 * m)
                elif iv.type == "rate_limit":
                    ceiling = baseline_lambda + (params["injected_lambda"] - baseline_lambda) * (1.0 - m)
                    lambda_eff = min(lambda_eff, ceiling)
                    current_lambda = min(current_lambda, ceiling)
                elif iv.type == "influencer_amplify":
                    beta_eff *= (1.0 + m)
                    if step == iv.start_step:  # one-off re-injection spike
                        current_lambda += params["injected_lambda"] * m
                        lambda_eff = current_lambda

            zeta = phi * max(0.0, lambda_eff - baseline_lambda)
            state = seir.step(dt, beta_eff, zeta)

            history.append({
                "t": seir.t,
                "S": state[0].item(), "E": state[1].item(), "I": state[2].item(),
                "R": state[3].item(), "Z": state[4].item(), "D": state[5].item(),
                "beta": round(beta_eff, 4),
                "zeta": round(zeta, 4),
                "lambda_val": round(lambda_eff, 4),
                "mean_opinion": 0.0,
            })

        return history

    # ------------------------------------------------------------------ #
    @staticmethod
    def metrics(history: List[Dict], N: float) -> TrajectoryMetrics:
        if not history:
            return TrajectoryMetrics()
        infected = [h["I"] for h in history]
        peak_I = max(infected)
        peak_step = int(np.argmax(infected))
        final = history[-1]
        return TrajectoryMetrics(
            peak_I=peak_I,
            peak_I_step=peak_step,
            total_reach=float(N) - final["S"],
            final_Z=final["Z"],
            final_D=final["D"],
            auc_I=float(np.trapz(infected)),
            final_mean_opinion=final.get("mean_opinion", 0.0),
        )

    def compare(
        self,
        params: Dict,
        interventions: List[Intervention],
        include_history: bool = True,
    ) -> InterventionReport:
        """Run baseline (no interventions) and treatment, return deltas."""
        baseline = self.run_trajectory(params, interventions=None)
        treatment = self.run_trajectory(params, interventions=interventions)

        bm = self.metrics(baseline, params["N"]).to_dict()
        tm = self.metrics(treatment, params["N"]).to_dict()

        deltas, pct = {}, {}
        for k in bm:
            if isinstance(bm[k], (int, float)):
                deltas[k] = round(tm[k] - bm[k], 4)
                pct[k] = round(100.0 * (tm[k] - bm[k]) / bm[k], 2) if bm[k] not in (0, 0.0) else 0.0

        return InterventionReport(
            baseline_metrics=bm,
            treatment_metrics=tm,
            deltas=deltas,
            pct_change=pct,
            interventions=[{"type": i.type, "start_step": i.start_step,
                            "magnitude": i.magnitude, "name": i.name} for i in interventions],
            baseline_history=baseline if include_history else [],
            treatment_history=treatment if include_history else [],
        )
