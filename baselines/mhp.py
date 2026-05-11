import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hawkes import StreamingHawkesLoss

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


class StaticHawkesBaseline(nn.Module):
    """
    Trainable static multivariate Hawkes baseline.

    Unlike the neural SimCity head, this model has no graph memory and no
    narrative/user embeddings. It learns one global background rate, one GDELT
    response weight, one cross-platform excitation matrix, and one decay matrix.
    That makes it a strong classical benchmark for proving the neural model
    adds value beyond platform-level cascade statistics.
    """

    def __init__(
        self,
        num_platforms=3,
        time_scale_seconds=3600.0,
        initial_mu=0.1,
        initial_alpha=0.05,
        initial_gamma=1.0,
    ):
        super().__init__()
        self.num_platforms = num_platforms
        self.hawkes_loss = StreamingHawkesLoss(
            num_platforms=num_platforms,
            time_scale_seconds=time_scale_seconds,
        )

        self.mu_logits = nn.Parameter(
            torch.full((num_platforms,), _inverse_softplus(initial_mu))
        )
        self.gdelt_logits = nn.Parameter(torch.full((num_platforms,), -2.0))
        self.alpha_logits = nn.Parameter(
            torch.full((num_platforms, num_platforms), _inverse_softplus(initial_alpha))
        )
        self.gamma_logits = nn.Parameter(
            torch.full((num_platforms, num_platforms), _inverse_softplus(initial_gamma))
        )

    def reset_state(self):
        self.hawkes_loss.reset_state()

    def detach_state(self):
        self.hawkes_loss.detach_state()

    def parameters_for_events(self, gdelt_volume):
        if gdelt_volume.dim() == 1:
            gdelt_volume = gdelt_volume.unsqueeze(-1)

        gdelt_signal = torch.log1p(gdelt_volume.clamp_min(0.0))
        mu_base = F.softplus(self.mu_logits).view(1, -1)
        gdelt_weight = F.softplus(self.gdelt_logits).view(1, -1)
        mu = mu_base + gdelt_signal * gdelt_weight + 1e-6

        batch_size = gdelt_volume.size(0)
        alpha = (
            F.softplus(self.alpha_logits)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        gamma = (
            F.softplus(self.gamma_logits)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            + 1e-6
        )
        return mu, alpha, gamma

    def forward(self, t_events, platform_ids, gdelt_volume, update_state=True):
        mu, alpha, gamma = self.parameters_for_events(gdelt_volume)
        return self.hawkes_loss(
            t_events,
            platform_ids,
            mu,
            alpha,
            gamma,
            update_state=update_state,
        )

    def learned_parameters(self):
        with torch.no_grad():
            return {
                "mu": F.softplus(self.mu_logits).detach().cpu(),
                "gdelt_weight": F.softplus(self.gdelt_logits).detach().cpu(),
                "alpha": F.softplus(self.alpha_logits).detach().cpu(),
                "gamma": (F.softplus(self.gamma_logits) + 1e-6).detach().cpu(),
            }


def _inverse_softplus(value):
    value = torch.as_tensor(value, dtype=torch.float)
    return torch.log(torch.expm1(value).clamp_min(1e-8))
