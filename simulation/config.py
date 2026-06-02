"""
Simulation configuration — centralizes all simulation parameters.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulationConfig:
    """Central configuration for all simulation parameters."""

    # === Population ===
    N: int = 100000
    initial_S: int = 90000
    initial_E: int = 1000
    initial_I: int = 0
    initial_R: int = 9000
    initial_Z: int = 0
    initial_D: int = 0

    # === SEIR-Z-D Parameters ===
    theta: float = 2.0        # Exposure rate
    sigma: float = 0.5        # E → I transition rate
    gamma_I: float = 0.1      # I → R recovery rate
    delta_D: float = 0.01     # I → D death/ban rate
    base_beta_macro: float = 0.8  # Base transmission rate

    # === Hawkes Process ===
    baseline_lambda: float = 1.0   # Background event rate
    injected_lambda: float = 50.0  # Injected cascade intensity
    decay_gamma: float = 0.2       # Hawkes decay rate

    # === Opinion Dynamics ===
    phi: float = 0.05         # Zombie recruitment rate
    deffuant_eta: float = 0.5
    deffuant_epsilon: float = 0.3
    deffuant_kappa: float = 10.0

    # === Simulation Control ===
    steps: int = 30
    dt: float = 1.0
    cache_dir: str = "simulation_cache"

    # === Agent Config ===
    agent_types: List[str] = field(default_factory=lambda: [
        "influencer", "bot", "skeptic", "community"
    ])
    max_agent_turns: int = 2

    @property
    def initial_state(self) -> list:
        return [
            self.initial_S, self.initial_E, self.initial_I,
            self.initial_R, self.initial_Z, self.initial_D,
        ]

    @classmethod
    def from_env(cls) -> "SimulationConfig":
        """Load config from environment variables."""
        return cls(
            N=int(os.getenv("SIM_N", "100000")),
            steps=int(os.getenv("SIM_STEPS", "30")),
            theta=float(os.getenv("SIM_THETA", "2.0")),
            sigma=float(os.getenv("SIM_SIGMA", "0.5")),
        )
