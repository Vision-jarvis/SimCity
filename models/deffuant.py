import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothDeffuant(nn.Module):
    """
    Filippov-Compliant Smooth Deffuant Model for opinion dynamics.
    Implements Section 5 of the SimCity architecture.
    """
    def __init__(self, eta=0.1, kappa=100.0, epsilon_base=0.5, alpha_r=0.5):
        """
        Args:
            eta: Opinion convergence rate
            kappa: Scaling constant for steepness rho
            epsilon_base: Base confidence bound
            alpha_r: Weighting factor between structural and historical radicalization
        """
        super().__init__()
        self.eta = eta
        self.kappa = kappa
        self.epsilon_base = epsilon_base
        self.alpha_r = alpha_r
        # rho = kappa / epsilon_base (Eq 22)
        self.rho = self.kappa / self.epsilon_base

    def compute_radicalization(self, total_edges, cross_community_edges, num_exposed, num_rejected):
        """
        Computes dynamic radicalization score (Eq 24).
        Args:
            total_edges: Tensor of shape (N,) containing |N_v|
            cross_community_edges: Tensor of shape (N,) containing |\partial N_v \cap C_v|
            num_exposed: Tensor of shape (N,) containing total exposure events
            num_rejected: Tensor of shape (N,) containing total rejected events
        Returns:
            Rad(v, t): Tensor of shape (N,) in [0, 1]
        """
        # Structural isolation: 1 if 0 cross-community edges, 0 if all are cross-community
        # Handle zero division safely
        struct_iso = torch.ones_like(total_edges, dtype=torch.float32)
        mask_edges = total_edges > 0
        struct_iso[mask_edges] = 1.0 - (cross_community_edges[mask_edges].float() / total_edges[mask_edges].float())
        
        # Historical rigidity
        hist_rigid = torch.zeros_like(num_exposed, dtype=torch.float32)
        mask_exp = num_exposed > 0
        hist_rigid[mask_exp] = num_rejected[mask_exp].float() / num_exposed[mask_exp].float()
        
        rad = self.alpha_r * struct_iso + (1.0 - self.alpha_r) * hist_rigid
        return rad.clamp(0.0, 1.0)

    def compute_epsilon_p(self, radicalization):
        """
        Computes the platform-specific confidence bound (Eq 23).
        """
        return self.epsilon_base * (1.0 - radicalization)

    def compute_derivative(self, x_v, x_c, epsilon_p):
        """
        Computes the continuous derivative dx_v(t)/dt (Eq 21).
        Args:
            x_v: Node opinion
            x_c: Content opinion
            epsilon_p: Node confidence bound
        """
        diff = torch.abs(x_v - x_c)
        # sigma(rho * (epsilon_p - |x_v - x_c|))
        # This replaces the hard indicator 1_{|x_v - x_c| <= epsilon_p} with a smooth logistic
        z = self.rho * (epsilon_p - diff)
        sigma = torch.sigmoid(z)
        
        # dx_v/dt = eta * (x_c - x_v) * sigma
        dx_dt = self.eta * (x_c - x_v) * sigma
        return dx_dt

    def forward(self, x_v, x_c, dt, total_edges, cross_community_edges, num_exposed, num_rejected):
        """
        Discrete forward update for a time step dt.
        Returns the new opinions x_v(t + dt) and a boolean tensor indicating if the event was rejected.
        """
        rad = self.compute_radicalization(total_edges, cross_community_edges, num_exposed, num_rejected)
        eps_p = self.compute_epsilon_p(rad)
        dx_dt = self.compute_derivative(x_v, x_c, eps_p)
        
        # Euler integration step
        x_v_new = x_v + dt * dx_dt
        
        # An event is strictly rejected if |x_v - x_c| > epsilon_p (used to update historical rigidity)
        rejected = torch.abs(x_v - x_c) > eps_p
        
        return x_v_new, rejected, rad
