"""Simulation API endpoints — scenario run and replay."""

from fastapi import APIRouter, HTTPException
import json
import os
import uuid
import torch

from api.schemas.simulation_schemas import ScenarioConfig, SimulationResult
from simulation.engine import SimCityEngine
from models.deffuant import SmoothDeffuant

router = APIRouter(prefix="/simulate", tags=["simulation"])

CACHE_DIR = "simulation_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@router.post("/run", response_model=SimulationResult)
async def run_simulation(config: ScenarioConfig):
    """Run a macroscopic SEIR-Z-D simulation."""
    try:
        engine = SimCityEngine(None, None, None, None, None)

        initial_state = [
            config.initial_S, config.initial_E, config.initial_I,
            config.initial_R, config.initial_Z, config.initial_D,
        ]

        engine.initialize_simulation(
            initial_state=initial_state,
            N=config.N,
            theta=config.theta,
            sigma=config.sigma,
            gamma_I=config.gamma_I,
            delta_D=config.delta_D,
        )

        deffuant = SmoothDeffuant(eta=0.5, epsilon_base=0.3, kappa=10.0)
        initial_opinions = torch.cat([
            torch.normal(0.2, 0.1, (config.N // 2,)),
            torch.normal(0.8, 0.1, (config.N // 2,)),
        ])

        history = engine.run_scenario(
            steps=config.steps,
            dt=config.dt,
            base_beta_macro=config.base_beta_macro,
            base_lambda=config.injected_lambda,
            baseline_lambda=config.baseline_lambda,
            decay_gamma=config.decay_gamma,
            phi=config.phi,
            deffuant_model=deffuant,
            initial_opinions=initial_opinions,
            content_opinion=0.9,
        )

        run_id = str(uuid.uuid4())
        cache_path = os.path.join(CACHE_DIR, f"{run_id}.json")

        result = SimulationResult(
            run_id=run_id,
            scenario=config,
            results=history,
        )

        with open(cache_path, "w") as f:
            json.dump(result.model_dump(), f)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/replay/{run_id}")
async def replay_simulation(run_id: str):
    """Fetch cached simulation results."""
    cache_path = os.path.join(CACHE_DIR, f"{run_id}.json")
    if not os.path.exists(cache_path):
        raise HTTPException(status_code=404, detail="Simulation run not found")

    with open(cache_path, "r") as f:
        data = json.load(f)

    return data


@router.get("/presets")
async def list_presets():
    """List available scenario presets."""
    return {
        "presets": [
            {
                "id": "misinfo_outbreak",
                "name": "Misinformation Outbreak",
                "description": "Coordinated bot network injects disinformation",
                "params": {"theta": 2.5, "sigma": 0.7, "injected_lambda": 51.0},
            },
            {
                "id": "influencer_tweet",
                "name": "Influencer Tweet Storm",
                "description": "Major influencer triggers cross-platform cascade",
                "params": {"theta": 2.0, "sigma": 0.5, "injected_lambda": 50.0},
            },
            {
                "id": "platform_outage",
                "name": "Platform Outage",
                "description": "Major platform goes offline, traffic migrates",
                "params": {"theta": 1.5, "sigma": 0.3, "injected_lambda": 30.0},
            },
        ]
    }
