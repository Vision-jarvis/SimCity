import torch
import numpy as np
from simulation.seir_z_d import StochasticSEIRZD

class SimCityEngine:
    """
    Simulation Engine that couples the macroscopic SEIR-Z-D model
    with the microscopic TGN and Hawkes models.
    """
    def __init__(self, tgn, influence_scorer, hmf_bridge, virality_head, hawkes_model, device='cpu'):
        self.tgn = tgn
        self.influence_scorer = influence_scorer
        self.hmf_bridge = hmf_bridge
        self.virality_head = virality_head
        self.hawkes_model = hawkes_model
        self.device = device

    def initialize_simulation(self, initial_state, N, theta, sigma, gamma_I, delta_D):
        self.seir = StochasticSEIRZD(
            initial_state=initial_state,
            N=N,
            theta=theta,
            sigma=sigma,
            gamma_I=gamma_I,
            delta_D=delta_D,
            device=self.device
        )
        self.history = []
        
    def get_beta_macro(self, h_src, h_dst, h_P, gdelt_volume, src_nodes):
        """
        Calculates macroscopic beta(t) via HMF using the trained Virality Head.
        """
        beta_micro, mu, alpha, gamma, _ = self.virality_head(h_src, h_dst, h_P, gdelt_volume)
        # Use HMF to aggregate micro to macro
        beta_macro = self.hmf_bridge(beta_micro, self.influence_scorer.temporal_degree[src_nodes])
        return beta_macro, mu, alpha, gamma

    def get_zeta(self, lambda_at_t, baseline_lambda, phi=1.0):
        """
        Eq 46: zeta(t) = phi * max(0, lambda(t) - baseline_lambda)
        """
        surge = max(0.0, lambda_at_t - baseline_lambda)
        return phi * surge

    def run_scenario(self, steps, dt, base_beta_macro, base_lambda, baseline_lambda, decay_gamma, phi=1.0, deffuant_model=None, initial_opinions=None, content_opinion=0.0):
        """
        Runs the simulation forward in time.
        In a full graph-coupled run, base_beta_macro would be updated by new TGN edges.
        Here we assume a static graph future where Hawkes intensity decays, 
        driving zeta(t) down, while SEIR dynamics unfold.
        If deffuant_model is provided, we simulate opinion shifts over time.
        """
        current_lambda = base_lambda
        opinions = initial_opinions.clone() if initial_opinions is not None else None
        
        for step in range(steps):
            # 1. Update Hawkes intensity (analytical decay)
            current_lambda = current_lambda * np.exp(-decay_gamma * dt)
            
            # 2. Compute Zeta
            zeta = self.get_zeta(current_lambda, baseline_lambda, phi)
            
            # 3. Step SEIR-Z-D Stochastic Model
            # We assume beta remains relatively constant if no new structural edges form
            state = self.seir.step(dt, base_beta_macro, zeta)
            
            # 4. Polarization Shift (Smooth Deffuant)
            mean_opinion = 0.0
            if deffuant_model is not None and opinions is not None:
                # We simulate a simplified global interaction with the content opinion
                # Assuming all nodes interact with the content rate proportional to beta
                # A full graph simulation would do this per-edge
                interaction_strength = base_beta_macro * dt
                # Mock radicalization scalar for population (e.g. 0.5)
                rad_scalar = torch.full_like(opinions, 0.5)
                eps_p = deffuant_model.compute_epsilon_p(rad_scalar)
                
                # Compute opinion derivatives
                d_opinions = deffuant_model.compute_derivative(opinions, content_opinion, eps_p)
                opinions = opinions + d_opinions * interaction_strength
                mean_opinion = opinions.mean().item()
            
            # Log
            self.history.append({
                't': self.seir.t,
                'S': state[0].item(),
                'E': state[1].item(),
                'I': state[2].item(),
                'R': state[3].item(),
                'Z': state[4].item(),
                'D': state[5].item(),
                'beta': base_beta_macro,
                'zeta': zeta,
                'lambda': current_lambda,
                'mean_opinion': mean_opinion
            })
            
        return self.history
