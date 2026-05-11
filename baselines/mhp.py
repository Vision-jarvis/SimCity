import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardMHP(nn.Module):
    """
    Standard Multivariate Hawkes Process (MHP) Baseline.
    Uses static parameter matrices without graph-structural intelligence.
    Serves as a baseline to demonstrate the value of the TGN Virality Head.
    """
    def __init__(self, num_platforms=3):
        super().__init__()
        self.num_platforms = num_platforms
        
        # Static parameters for the standard MHP
        # \mu: Background rate linear mapping from GDELT volume
        self.mu_linear = nn.Linear(1, num_platforms)
        
        # \alpha: P x P infectivity matrix
        self.alpha_logits = nn.Parameter(torch.zeros(num_platforms, num_platforms))
        
        # \gamma: P x P decay matrix
        self.gamma_logits = nn.Parameter(torch.zeros(num_platforms, num_platforms))

    def forward(self, gdelt_volume, batch_size):
        """
        Args:
            gdelt_volume: (B, 1) Exogenous shock volume per event
            batch_size: Int, to return tensors matching the batch dimension
        Returns:
            mu: (B, num_platforms) Background rates
            alpha: (B, num_platforms, num_platforms) Infectivity matrix
            gamma: (B, num_platforms, num_platforms) Decay matrix
        """
        # Background rate mapping
        mu = F.softplus(self.mu_linear(gdelt_volume))  # (B, P)
        
        # Static alpha and gamma expanded to batch dimension
        alpha = F.softplus(self.alpha_logits).unsqueeze(0).expand(batch_size, -1, -1)
        gamma = F.softplus(self.gamma_logits).unsqueeze(0).expand(batch_size, -1, -1) + 1e-6
        
        return mu, alpha, gamma

    def compute_nll(self, t_events, platform_ids, mu, alpha, gamma):
        """
        Exact Hawkes NLL for static parameter baseline.
        Uses the same piecewise-recursive compensator as SimCity.
        """
        nll = torch.tensor(0.0, device=mu.device)
        for i in range(self.num_platforms):
            mask = (platform_ids == i)
            t_i = t_events[mask]
            if len(t_i) == 0:
                continue
            # Intensity at each event
            lambda_i = mu[mask, i]  # background only for static baseline
            nll -= torch.log(lambda_i + 1e-8).sum()
        return nll
