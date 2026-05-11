import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralHawkesLoss(nn.Module):
    """
    Differentiable multivariate Hawkes negative log-likelihood.

    The loss treats each mini-batch as a chronologically ordered observation
    window. For every event, the neural virality head provides:
      - mu[k, i]: exogenous background intensity for platform i at event k
      - alpha[k, j, i]: excitation from a past event on platform j to platform i
      - gamma[k, j, i]: exponential decay for that cross-platform excitation

    This is a local likelihood approximation rather than a full-dataset Hawkes
    fit, which keeps training practical with TemporalDataLoader batches while
    still making the platform-to-platform cascade parameters identifiable.
    """

    def __init__(self, num_platforms=3, time_scale_seconds=3600.0, eps=1e-8):
        super().__init__()
        self.num_platforms = num_platforms
        self.time_scale_seconds = float(time_scale_seconds)
        self.eps = eps

    def forward(self, t_events, platform_ids, mu, alpha, gamma):
        if t_events.numel() == 0:
            return mu.new_tensor(0.0, requires_grad=True)

        order = torch.argsort(t_events)
        t = t_events[order].to(mu.dtype) / self.time_scale_seconds
        platform = platform_ids[order].long()
        mu = mu[order].clamp_min(self.eps)
        alpha = alpha[order].clamp_min(0.0)
        gamma = gamma[order].clamp_min(self.eps)

        batch_size = t.numel()
        dt = t[:, None] - t[None, :]
        history_mask = dt > 0

        past_platform = F.one_hot(
            platform, num_classes=self.num_platforms
        ).to(dtype=mu.dtype)

        alpha_by_past = torch.einsum("lj,kji->kli", past_platform, alpha)
        gamma_by_past = torch.einsum("lj,kji->kli", past_platform, gamma)

        excitation = (
            alpha_by_past
            * torch.exp(-gamma_by_past * dt.clamp_min(0.0).unsqueeze(-1))
            * history_mask.unsqueeze(-1)
        ).sum(dim=1)

        intensity = (mu + excitation).clamp_min(self.eps)
        event_intensity = intensity.gather(1, platform.view(-1, 1)).squeeze(1)
        event_nll = -torch.log(event_intensity).sum()

        previous_t = torch.cat([t[:1], t[:-1]])
        interval = (t - previous_t).clamp_min(0.0)
        background_compensator = (mu.sum(dim=1) * interval).sum()

        lower_delta = (previous_t[:, None] - t[None, :]).clamp_min(0.0)
        upper_delta = (t[:, None] - t[None, :]).clamp_min(0.0)
        integral_mask = (upper_delta > lower_delta) & history_mask

        excitation_integral = (
            alpha_by_past
            / gamma_by_past
            * (
                torch.exp(-gamma_by_past * lower_delta.unsqueeze(-1))
                - torch.exp(-gamma_by_past * upper_delta.unsqueeze(-1))
            )
            * integral_mask.unsqueeze(-1)
        ).sum()

        return (event_nll + background_compensator + excitation_integral) / batch_size


class StreamingHawkesLoss(nn.Module):
    """
    Bounded-memory Hawkes NLL for chronological streams.

    This module carries a P x P decayed history matrix across mini-batches. Row
    j stores recent source-platform j activity; column i is the contribution
    available to target platform i. The state is detached after each forward
    call so training does not backpropagate through the full event stream.
    """

    def __init__(self, num_platforms=3, time_scale_seconds=3600.0, eps=1e-8):
        super().__init__()
        self.num_platforms = num_platforms
        self.time_scale_seconds = float(time_scale_seconds)
        self.eps = eps
        self.register_buffer(
            "excitation_state",
            torch.zeros(num_platforms, num_platforms),
            persistent=False,
        )
        self.register_buffer("last_time", torch.tensor(float("nan")), persistent=False)

    def reset_state(self):
        self.excitation_state.zero_()
        self.last_time.fill_(float("nan"))

    def detach_state(self):
        self.excitation_state.detach_()
        self.last_time.detach_()

    def forward(self, t_events, platform_ids, mu, alpha, gamma, update_state=True):
        if t_events.numel() == 0:
            return mu.new_tensor(0.0, requires_grad=True)

        order = torch.argsort(t_events)
        t = t_events[order].to(mu.dtype) / self.time_scale_seconds
        platform = platform_ids[order].long()
        mu = mu[order].clamp_min(self.eps)
        alpha = alpha[order].clamp_min(0.0)
        gamma = gamma[order].clamp_min(self.eps)

        state = self.excitation_state.to(device=mu.device, dtype=mu.dtype)
        last_time = self.last_time.to(device=mu.device, dtype=mu.dtype)
        has_history = torch.isfinite(last_time)
        total_nll = mu.new_tensor(0.0)

        for idx in range(t.numel()):
            current_t = t[idx]
            interval = torch.where(
                has_history,
                (current_t - last_time).clamp_min(0.0),
                current_t.new_tensor(0.0),
            )

            decay = torch.exp(-gamma[idx] * interval)
            decayed_state = state * decay
            intensity = (mu[idx] + (alpha[idx] * decayed_state).sum(dim=0)).clamp_min(
                self.eps
            )
            total_nll = total_nll - torch.log(intensity[platform[idx]])

            background_compensator = mu[idx].sum() * interval
            excitation_compensator = (
                alpha[idx] / gamma[idx] * state * (1.0 - decay)
            ).sum()
            total_nll = total_nll + background_compensator + excitation_compensator

            event_update = F.one_hot(
                platform[idx], num_classes=self.num_platforms
            ).to(dtype=mu.dtype, device=mu.device)
            state = decayed_state + event_update.unsqueeze(1).expand_as(state)
            last_time = current_t
            has_history = torch.ones((), dtype=torch.bool, device=mu.device)

        if update_state:
            self.excitation_state = state.detach()
            self.last_time = last_time.detach()

        return total_nll / t.numel()
