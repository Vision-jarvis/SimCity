import torch
import torch.nn as nn
import torch.nn.functional as F

class PlatformPooling(nn.Module):
    def __init__(self, embedding_dim, num_platforms, temperature=1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(temperature))
        self.num_platforms = num_platforms
        self.embedding_dim = embedding_dim

    def forward(self, h_nodes, platform_ids, influence_scores):
        """
        h_nodes: (N, embedding_dim)
        platform_ids: (N,) int tensor — which platform each node belongs to
        influence_scores: (N,) float tensor — I(v,t) for attention weighting
        Returns: (num_platforms, embedding_dim)
        """
        platform_embeddings = torch.zeros(
            self.num_platforms, self.embedding_dim, device=h_nodes.device
        )
        for p in range(self.num_platforms):
            mask = (platform_ids == p)
            if mask.sum() == 0:
                continue
            scores = influence_scores[mask]
            weights = torch.softmax(self.tau * scores, dim=0)
            platform_embeddings[p] = (weights.unsqueeze(1) * h_nodes[mask]).sum(0)
        return platform_embeddings

class ViralityHead(nn.Module):
    """
    Neural mapping from TGN embeddings to SEIR-Z-D epidemiological 
    and Multivariate Hawkes process parameters.
    """
    def __init__(self, embedding_dim, num_platforms=3, hidden_dim=64, exc_dim=0):
        super().__init__()
        self.num_platforms = num_platforms
        # Number of observed excitation features (e.g. current cross-platform
        # Hawkes intensity / recent event-rate) fed into the virality readout.
        # Near-future engagement is governed by the instantaneous excitation
        # state; without it the head is blind to the signal it must predict.
        self.exc_dim = exc_dim
        
        # MLP for beta (infectivity/exposure rate)
        # Input: [h_v(t) || h_c(t)] -> Dim: 2 * embedding_dim
        self.mlp_beta = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # MLPs for Hawkes parameters (alpha and gamma)
        # Input: [flatten(h_P) || h_c(t)] -> Dim: (num_platforms + 1) * embedding_dim
        hawkes_in_dim = (num_platforms + 1) * embedding_dim
        self.mlp_mu = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_platforms)
        )

        self.mlp_alpha = nn.Sequential(
            nn.Linear(hawkes_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_platforms * num_platforms)
        )
        
        self.mlp_gamma = nn.Sequential(
            nn.Linear(hawkes_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_platforms * num_platforms)
        )
        
        # Virality Regression Output (for E_c prediction). Input is
        # [h_v || h_c || mean(h_P)] plus any observed excitation features.
        self.mlp_virality = nn.Sequential(
            nn.Linear(3 * embedding_dim + exc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Compute target mean/std from training data once and register as buffers
        self.register_buffer('target_mean', torch.tensor(0.0))
        self.register_buffer('target_std', torch.tensor(1.0))

    def forward(self, h_v, h_c, h_P, gdelt_volume=None, exc_feats=None):
        """
        Args:
            h_v: (B, D) Source node (user) temporal embedding
            h_c: (B, D) Destination node (narrative) temporal embedding
            h_P: (num_platforms, D) Platform pooled embeddings
            gdelt_volume: optional (B,) or (B, 1) exogenous news volume
            exc_feats: optional (B, exc_dim) observed excitation features
                (current cross-platform intensity / recent event rate)
        """
        B = h_v.size(0)
        if gdelt_volume is None:
            gdelt_volume = h_v.new_zeros(B, 1)
        elif gdelt_volume.dim() == 1:
            gdelt_volume = gdelt_volume.unsqueeze(-1)
        gdelt_signal = torch.log1p(gdelt_volume.clamp_min(0.0))
        
        # 1. SEIR and Virality (Event-level)
        z_event = torch.cat([h_v, h_c], dim=-1)
        
        beta = F.softplus(self.mlp_beta(z_event)).squeeze(-1)
        
        # Flatten platform embeddings: (P, D) -> (P * D) -> (1, P * D) -> (B, P * D)
        h_P_flat = h_P.view(1, -1).expand(B, -1)
        z_hawkes = torch.cat([h_P_flat, h_c], dim=-1)
        
        mu = F.softplus(self.mlp_mu(torch.cat([h_c, gdelt_signal], dim=-1))) + 1e-6
        alpha = F.softplus(self.mlp_alpha(z_hawkes)).view(-1, self.num_platforms, self.num_platforms)
        gamma = F.softplus(self.mlp_gamma(z_hawkes)).view(-1, self.num_platforms, self.num_platforms) + 1e-6
        
        z_virality = torch.cat([h_v, h_c, h_P.mean(dim=0).unsqueeze(0).expand(B, -1)], dim=-1)
        if self.exc_dim > 0:
            if exc_feats is None:
                exc_feats = z_virality.new_zeros(B, self.exc_dim)
            z_virality = torch.cat([z_virality, exc_feats], dim=-1)
        virality = self.mlp_virality(z_virality).squeeze(-1)
        virality = virality * self.target_std + self.target_mean  # scale to target range
        
        return beta, mu, alpha, gamma, virality
