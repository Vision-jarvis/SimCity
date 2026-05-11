import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCityLoss(nn.Module):
    """
    Homoscedastic Multi-Task Loss for the SimCity architecture.
    Balances TGN InfoNCE, Hawkes NLL, and Virality Regression
    using Kendall's uncertainty weighting (Kendall et al., 2018).
    """
    def __init__(self):
        super().__init__()
        # Learnable log variances to ensure positivity and stable gradients
        # We initialize them to 0.0, meaning sigma^2 = 1.0 initially.
        self.log_vars = nn.Parameter(torch.zeros(3))

    def compute_tgn_loss(self, h_src, h_dst_pos, h_dst_neg):
        """
        InfoNCE Loss for TGN temporal edge prediction.
        h_src:     (B, D)
        h_dst_pos: (B, D)
        h_dst_neg: (B, M, D)  — M negatives per positive
        """
        # Positive scores: (B, 1)
        pos = (h_src * h_dst_pos).sum(-1, keepdim=True)
        # Negative scores: (B, M)
        neg = torch.bmm(h_dst_neg, h_src.unsqueeze(-1)).squeeze(-1)
        # InfoNCE: softmax over [pos, neg1, ..., negM]
        logits = torch.cat([pos, neg], dim=1)  # (B, M+1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute_virality_loss(self, pred_virality, true_virality):
        """
        MSE on log-engagement (L_Virality).
        """
        mask = ~torch.isnan(true_virality)
        if not mask.any():
            return torch.tensor(0.0, device=pred_virality.device, requires_grad=True)
        return F.mse_loss(pred_virality[mask], true_virality[mask])

    def compute_hawkes_loss(self, lambda_at_events, compensator):
        """
        Real Hawkes NLL: -sum(log lambda(t_k)) + compensator integral
        lambda_at_events: (num_events,) — intensity evaluated at each event time
        compensator: (num_events,) — piecewise integral contribution per event interval
        """
        nll = -torch.log(lambda_at_events + 1e-8).mean() + compensator.mean()
        return nll

    def forward(self, l_tgn, l_hawkes, l_virality):
        """
        Combines task losses using homoscedastic uncertainty weighting.
        """
        losses = torch.stack([l_tgn, l_hawkes, l_virality])
        log_vars = self.log_vars.clamp(min=-6.0, max=6.0)
        precision = torch.exp(-log_vars)
        return (precision * losses + 0.5 * log_vars).sum()
