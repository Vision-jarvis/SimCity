from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import os
import uuid
from typing import Dict, Any, List

from simulation.engine import SimCityEngine
from models.deffuant import SmoothDeffuant
import torch

app = FastAPI(title="SimCity Simulator API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache directory for simulation results
CACHE_DIR = "simulation_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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

@app.post("/simulate/run")
async def run_simulation(config: ScenarioConfig):
    """
    Runs a macroscopic SEIR-Z-D simulation based on the scenario config.
    Tracks population curves and polarization delta.
    """
    try:
        engine = SimCityEngine(None, None, None, None, None)
        
        initial_state = [
            config.initial_S, config.initial_E, config.initial_I,
            config.initial_R, config.initial_Z, config.initial_D
        ]
        
        engine.initialize_simulation(
            initial_state=initial_state,
            N=config.N,
            theta=config.theta,
            sigma=config.sigma,
            gamma_I=config.gamma_I,
            delta_D=config.delta_D
        )
        
        # Initialize Deffuant for polarization tracking
        deffuant = SmoothDeffuant(eta=0.5, epsilon_base=0.3, kappa=10.0)
        # Mock initial opinions clustered around 0.2 and 0.8
        initial_opinions = torch.cat([torch.normal(0.2, 0.1, (config.N//2,)), 
                                      torch.normal(0.8, 0.1, (config.N//2,))])
        content_opinion = 0.9 # Extreme content
        
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
            content_opinion=content_opinion
        )
        
        # Cache results
        run_id = str(uuid.uuid4())
        cache_path = os.path.join(CACHE_DIR, f"{run_id}.json")
        
        result_payload = {
            "run_id": run_id,
            "scenario": config.dict(),
            "results": history
        }
        
        with open(cache_path, 'w') as f:
            json.dump(result_payload, f)
            
        return result_payload
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/simulate/replay/{run_id}")
async def replay_simulation(run_id: str):
    """
    Fetches cached simulation results by run_id.
    """
    cache_path = os.path.join(CACHE_DIR, f"{run_id}.json")
    if not os.path.exists(cache_path):
        raise HTTPException(status_code=404, detail="Simulation run not found")
        
    with open(cache_path, 'r') as f:
        data = json.load(f)
        
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
