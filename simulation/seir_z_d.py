import torch
import numpy as np

class StochasticSEIRZD:
    """
    Stochastic Master Equation (SME) solver for the SEIR-Z-D compartmental model
    using the Gillespie Stochastic Simulation Algorithm (SSA).
    
    Equations refer to Section 1.2 of main.tex.
    State vector: n(t) = (S, E, I, R, Z, D)
    """
    def __init__(self, initial_state, N, theta, sigma, gamma_I, delta_D, device='cpu'):
        """
        initial_state: tuple/list of 6 integers (S, E, I, R, Z, D)
        N: Total population (must equal sum of initial_state)
        theta: Algorithmic boost factor for Zombie content
        sigma: E -> I activation rate
        gamma_I: I -> R decay rate
        delta_D: Z -> D archival rate
        """
        assert sum(initial_state) == N, "Initial state sum must equal N"
        self.state = torch.tensor(initial_state, dtype=torch.float32, device=device)
        self.N = float(N)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.gamma_I = float(gamma_I)
        self.delta_D = float(delta_D)
        self.device = device
        
        # Jump vectors
        self.r = torch.tensor([
            [-1,  1,  0,  0,  0,  0],  # T1: S -> E
            [ 0, -1,  1,  0,  0,  0],  # T2: E -> I
            [ 0,  0, -1,  1,  0,  0],  # T3: I -> R
            [ 0,  0,  0, -1,  1,  0],  # T4: R -> Z
            [ 0,  0,  0,  0, -1,  1],  # T5: Z -> D
        ], dtype=torch.float32, device=device)
        
        self.t = 0.0

    def compute_rates(self, beta, zeta):
        """
        Computes the 5 transition rates W_k(n)
        beta: Macroscopic exposure rate at time t
        zeta: Zombie resurgence rate at time t
        """
        n_S, n_E, n_I, n_R, n_Z, n_D = self.state
        
        W = torch.zeros(5, device=self.device)
        # T1: Exposure
        W[0] = beta * (n_S * (n_I + self.theta * n_Z)) / self.N
        # T2: Activation
        W[1] = self.sigma * n_E
        # T3: Decay
        W[2] = self.gamma_I * n_I
        # T4: Resurgence
        W[3] = zeta * n_R
        # T5: Archival
        W[4] = self.delta_D * n_Z
        
        # Ensure non-negative due to float drift
        return torch.clamp(W, min=0.0)

    def step(self, dt, beta, zeta):
        """
        Advances the state by dt using the Gillespie algorithm, assuming
        beta and zeta are constant over this small dt window.
        Returns the new state.
        """
        target_t = self.t + dt
        
        while self.t < target_t:
            rates = self.compute_rates(beta, zeta)
            total_rate = rates.sum().item()
            
            if total_rate <= 0:
                # Absorbing state or no transitions possible
                self.t = target_t
                break
                
            # Time to next reaction
            tau = np.random.exponential(scale=1.0 / total_rate)
            
            if self.t + tau > target_t:
                # Next reaction happens after our target time window
                self.t = target_t
                break
                
            # Determine which reaction occurs
            probs = (rates / total_rate).cpu().numpy()
            event_idx = np.random.choice(5, p=probs)
            
            # Apply jump
            self.state += self.r[event_idx]
            self.t += tau
            
            # Sanity check: prevent negative compartments due to rounding/drift
            self.state = torch.clamp(self.state, min=0.0)
            
        return self.state.clone()
