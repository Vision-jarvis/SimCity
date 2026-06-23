import torch
from simulation.engine import SimCityEngine
from models.deffuant import SmoothDeffuant

def test_engine_initialization():
    engine = SimCityEngine(None, None, None, None, None)
    initial_state = [900, 10, 0, 90, 0, 0]
    
    engine.initialize_simulation(
        initial_state=initial_state,
        N=1000,
        theta=1.0,
        sigma=0.5,
        gamma_I=0.1,
        delta_D=0.01
    )
    
    assert engine.seir is not None
    assert engine.seir.N == 1000

def test_engine_scenario_run():
    engine = SimCityEngine(None, None, None, None, None)
    initial_state = [900, 10, 0, 90, 0, 0]
    
    engine.initialize_simulation(
        initial_state=initial_state,
        N=1000,
        theta=1.0,
        sigma=0.5,
        gamma_I=0.1,
        delta_D=0.01
    )
    
    # Run a tiny scenario
    history = engine.run_scenario(
        steps=3,
        dt=1.0,
        base_beta_macro=0.5,
        base_lambda=5.0,
        baseline_lambda=1.0,
        decay_gamma=0.1,
        phi=0.1
    )
    
    assert len(history) == 3
    assert 't' in history[0]
    assert 'Z' in history[0]
    assert 'lambda' in history[0]
    
    # Check that lambda decays
    assert history[0]['lambda'] > history[2]['lambda']

def test_engine_polarization_tracking():
    engine = SimCityEngine(None, None, None, None, None)
    engine.initialize_simulation([900, 10, 0, 90, 0, 0], N=1000, theta=1.0, sigma=0.5, gamma_I=0.1, delta_D=0.01)
    
    deffuant = SmoothDeffuant(eta=0.5, epsilon_base=0.3, kappa=10.0)
    initial_opinions = torch.tensor([0.1, 0.9])
    content_opinion = 0.5
    
    history = engine.run_scenario(
        steps=2,
        dt=1.0,
        base_beta_macro=0.5,
        base_lambda=5.0,
        baseline_lambda=1.0,
        decay_gamma=0.1,
        phi=0.1,
        deffuant_model=deffuant,
        initial_opinions=initial_opinions,
        content_opinion=content_opinion
    )
    
    assert 'mean_opinion' in history[0]
    # mean_opinion should shift towards 0.5
    assert 0.0 < history[-1]['mean_opinion'] < 1.0
