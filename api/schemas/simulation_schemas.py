"""Pydantic schemas for simulation API endpoints."""

from pydantic import BaseModel
from typing import List, Dict, Any


class ScenarioConfig(BaseModel):
    scenario_id: str = "custom"
    N: int = 100000
    initial_S: int = 80000
    initial_E: int = 1000
    initial_I: int = 0
    initial_R: int = 19000
    initial_Z: int = 0
    initial_D: int = 0
    theta: float = 2.0
    sigma: float = 0.5
    gamma_I: float = 0.1
    delta_D: float = 0.01
    base_beta_macro: float = 0.8
    baseline_lambda: float = 1.0
    injected_lambda: float = 50.0
    decay_gamma: float = 0.2
    phi: float = 0.05
    steps: int = 30
    dt: float = 1.0


class SimulationStep(BaseModel):
    t: float
    S: float
    E: float
    I: float
    R: float
    Z: float
    D: float
    beta: float
    zeta: float
    lambda_val: float = 0.0
    mean_opinion: float = 0.0


class SimulationResult(BaseModel):
    run_id: str
    scenario: ScenarioConfig
    results: List[Dict[str, Any]]


class InterventionSpec(BaseModel):
    type: str  # fact_check | counter_narrative | deplatform_bots | rate_limit | influencer_amplify
    start_step: int = 0
    magnitude: float = 0.5
    name: str = ""


class InterventionRequest(BaseModel):
    scenario: ScenarioConfig = ScenarioConfig()
    interventions: List[InterventionSpec] = []
    include_history: bool = True


class InterventionResult(BaseModel):
    run_id: str
    baseline_metrics: Dict[str, Any]
    treatment_metrics: Dict[str, Any]
    deltas: Dict[str, Any]
    pct_change: Dict[str, Any]
    interventions: List[Dict[str, Any]]
    baseline_history: List[Dict[str, Any]] = []
    treatment_history: List[Dict[str, Any]] = []


class AgentMessage(BaseModel):
    name: str
    content: str
    role: str = ""


class AgentSimulationRequest(BaseModel):
    seed_message: str
    max_turns: int = 2
    agents: List[str] = ["influencer", "bot", "skeptic", "community"]
