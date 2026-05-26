import pytest
import torch
from simulation.seir_z_d import StochasticSEIRZD

def test_seir_zd_conservation():
    N = 1000
    initial_state = [990, 0, 10, 0, 0, 0]
    model = StochasticSEIRZD(
        initial_state=initial_state,
        N=N,
        theta=1.5,
        sigma=0.1,
        gamma_I=0.05,
        delta_D=0.01
    )
    
    beta = 0.5
    zeta = 0.02
    
    for _ in range(100):
        state = model.step(dt=1.0, beta=beta, zeta=zeta)
        # Check conservation N = sum(state)
        assert torch.allclose(state.sum(), torch.tensor(float(N)))
        # Check no negative compartments
        assert (state >= 0).all()

def test_seir_zd_absorbing_state():
    N = 1000
    initial_state = [0, 0, 0, 0, 0, 1000] # All Dead
    model = StochasticSEIRZD(
        initial_state=initial_state,
        N=N,
        theta=1.5,
        sigma=0.1,
        gamma_I=0.05,
        delta_D=0.01
    )
    
    state = model.step(dt=10.0, beta=1.0, zeta=1.0)
    # Shouldn't move
    assert state[5].item() == 1000
    
def test_seir_zd_zombie_resurgence():
    N = 1000
    initial_state = [0, 0, 0, 1000, 0, 0] # All Recovered
    model = StochasticSEIRZD(
        initial_state=initial_state,
        N=N,
        theta=1.5,
        sigma=0.1,
        gamma_I=0.05,
        delta_D=0.00 # Don't let them die
    )
    
    # Trigger resurgence
    model.step(dt=5.0, beta=0.0, zeta=1.0)
    
    # Should have zombies now
    state = model.state
    assert state[4].item() > 0 # Zombies > 0
    assert state[3].item() < 1000 # Recovered decreased
    assert torch.allclose(state.sum(), torch.tensor(float(N)))
