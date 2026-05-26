import torch
import torch.nn as nn

class DegreeCorrelatedHMF(nn.Module):
    """
    Degree-Correlated Heterogeneous Mean-Field (HMF) Bridge.
    Implements Section 4 of the SimCity architecture.
    Maps microscopic TGN infectivity predictions to macroscopic SEIR-Z exposure rates.
    """
    def __init__(self, max_degree_bin=100):
        super().__init__()
        self.max_degree_bin = max_degree_bin
        
        # P(k' | k) conditional degree distribution matrix
        # Shape: (max_degree_bin + 1, max_degree_bin + 1)
        # Entry [k, k'] is the probability a node of degree k connects to degree k'
        self.register_buffer('P_k_given_k', torch.eye(max_degree_bin + 1))
        
        # Track edge counts for online estimation of P(k' | k)
        self.register_buffer('edge_counts', torch.zeros(max_degree_bin + 1, max_degree_bin + 1))

    def update_degree_distribution(self, k_u, k_v):
        """
        Updates the static estimation of P(k' | k) using observed edges (u,v).
        Args:
            k_u: (B,) Temporal degree of source nodes
            k_v: (B,) Temporal degree of destination nodes
        """
        # Bin degrees
        k_u_bin = torch.clamp(k_u.long(), 0, self.max_degree_bin)
        k_v_bin = torch.clamp(k_v.long(), 0, self.max_degree_bin)
        
        # Increment counts
        # (k_u, k_v) and (k_v, k_u) for undirected mixing assumption
        indices = torch.stack([
            torch.cat([k_u_bin, k_v_bin]),
            torch.cat([k_v_bin, k_u_bin])
        ], dim=0)
        
        ones = torch.ones(indices.size(1), device=k_u.device)
        self.edge_counts.index_put_(tuple(indices), ones, accumulate=True)
        
        # Normalize rows to get probabilities
        row_sums = self.edge_counts.sum(dim=1, keepdim=True)
        # Avoid division by zero
        mask = row_sums > 0
        
        new_P = self.P_k_given_k.clone()
        new_P[mask.squeeze(-1)] = self.edge_counts[mask.squeeze(-1)] / row_sums[mask.squeeze(-1)]
        
        self.P_k_given_k = new_P

    def forward(self, beta_hat, k_v):
        """
        Computes the macroscopic beta(t) from node-level beta_hat_v(t).
        Eq 19: beta(t) = sum_k sum_k' P(k'|k) * k' * E_{v in I_k'}[beta_hat_v]
        
        Args:
            beta_hat: (B,) Predicted microscopic infectivities for the active (infected) nodes
            k_v: (B,) Effective temporal degree for the active nodes
            
        Returns:
            beta_macro: Scalar tensor representing the macroscopic exposure rate.
        """
        if beta_hat.numel() == 0:
            return beta_hat.new_tensor(0.0)
            
        k_v_bin = torch.clamp(k_v.long(), 0, self.max_degree_bin)
        
        # Compute E[beta_hat_v] for each degree bin k'
        # We scatter_mean beta_hat into bins based on k_v_bin
        sum_beta = torch.zeros(self.max_degree_bin + 1, device=beta_hat.device)
        count_beta = torch.zeros(self.max_degree_bin + 1, device=beta_hat.device)
        
        sum_beta.index_put_((k_v_bin,), beta_hat, accumulate=True)
        count_beta.index_put_((k_v_bin,), torch.ones_like(beta_hat), accumulate=True)
        
        # Avoid div by zero
        E_beta = torch.where(count_beta > 0, sum_beta / count_beta, torch.zeros_like(sum_beta))
        
        # Vectorized Eq 19 computation:
        # sum_k sum_k' P(k'|k) * k' * E[beta]
        # P(k'|k) has shape (k, k')
        # We want to sum over k and k'. 
        # Actually, in a macroscopic model, we typically average over the degree distribution of the susceptible population.
        # Assuming a uniform distribution over susceptible degrees k for this summation:
        
        # k_prime vector: (k',)
        k_prime = torch.arange(self.max_degree_bin + 1, device=beta_hat.device, dtype=beta_hat.dtype)
        
        # Inner sum over k': for a given k, sum_{k'} P(k'|k) * k' * E_beta[k']
        # P(k'|k) * k' * E_beta -> (k, k')
        inner_term = self.P_k_given_k * k_prime.unsqueeze(0) * E_beta.unsqueeze(0)
        
        # Sum over k' (dim=1)
        inner_sum = inner_term.sum(dim=1) # Shape: (k,)
        
        # Sum over k. (Assuming uniform density of susceptible degrees in the absence of full population state)
        # Normalizing by max_degree_bin to keep scale stable.
        beta_macro = inner_sum.mean()
        
        return beta_macro
