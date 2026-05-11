from collections import deque

import torch
import torch.nn.functional as F


class HawkesCascadeMonitor:
    """
    Event-level residual monitor for multivariate Hawkes streams.

    The monitor mirrors StreamingHawkesLoss state updates, but returns
    operational signals instead of a scalar loss. High event NLL means an event
    arrived with low model-assigned likelihood; high excitation mass means the
    stream is already in an amplified cascade state.
    """

    def __init__(
        self,
        num_platforms=3,
        time_scale_seconds=3600.0,
        rolling_window=128,
        warning_z=2.0,
        critical_z=3.0,
        eps=1e-8,
    ):
        self.num_platforms = num_platforms
        self.time_scale_seconds = float(time_scale_seconds)
        self.rolling_window = int(rolling_window)
        self.warning_z = float(warning_z)
        self.critical_z = float(critical_z)
        self.eps = eps
        self.reset_state()

    def reset_state(self):
        self.excitation_state = torch.zeros(self.num_platforms, self.num_platforms)
        self.last_time = torch.tensor(float("nan"))
        self._recent_nll = deque(maxlen=max(self.rolling_window, 2))

    def score(self, t_events, platform_ids, mu, alpha, gamma, update_state=True):
        if t_events.numel() == 0:
            return {
                "event_index": torch.empty(0, dtype=torch.long),
                "t": t_events,
                "platform": platform_ids,
                "event_intensity": t_events.new_empty(0),
                "total_intensity": t_events.new_empty(0),
                "compensator": t_events.new_empty(0),
                "event_nll": t_events.new_empty(0),
                "excitation_mass": t_events.new_empty(0),
                "rolling_z": t_events.new_empty(0),
                "alert_level": torch.empty(0, dtype=torch.long, device=t_events.device),
            }

        with torch.no_grad():
            order = torch.argsort(t_events)
            t = t_events[order].to(mu.dtype) / self.time_scale_seconds
            platform = platform_ids[order].long()
            mu = mu[order].clamp_min(self.eps)
            alpha = alpha[order].clamp_min(0.0)
            gamma = gamma[order].clamp_min(self.eps)

            state = self.excitation_state.to(device=mu.device, dtype=mu.dtype)
            last_time = self.last_time.to(device=mu.device, dtype=mu.dtype)
            has_history = torch.isfinite(last_time)

            event_intensities = []
            total_intensities = []
            compensators = []
            event_nlls = []
            excitation_masses = []
            rolling_z = []
            alert_levels = []

            for idx in range(t.numel()):
                current_t = t[idx]
                interval = torch.where(
                    has_history,
                    (current_t - last_time).clamp_min(0.0),
                    current_t.new_tensor(0.0),
                )

                decay = torch.exp(-gamma[idx] * interval)
                decayed_state = state * decay
                excitation = (alpha[idx] * decayed_state).sum(dim=0)
                intensity = (mu[idx] + excitation).clamp_min(self.eps)
                event_intensity = intensity[platform[idx]]

                background_compensator = mu[idx].sum() * interval
                excitation_compensator = (
                    alpha[idx] / gamma[idx] * state * (1.0 - decay)
                ).sum()
                compensator = background_compensator + excitation_compensator
                event_nll = -torch.log(event_intensity) + compensator
                excitation_mass = excitation.sum()

                z_value = self._rolling_z(float(event_nll.detach().cpu()))
                if z_value >= self.critical_z:
                    level = 2
                elif z_value >= self.warning_z:
                    level = 1
                else:
                    level = 0

                event_intensities.append(event_intensity)
                total_intensities.append(intensity.sum())
                compensators.append(compensator)
                event_nlls.append(event_nll)
                excitation_masses.append(excitation_mass)
                rolling_z.append(mu.new_tensor(z_value))
                alert_levels.append(torch.tensor(level, dtype=torch.long, device=mu.device))

                self._recent_nll.append(float(event_nll.detach().cpu()))
                event_update = F.one_hot(
                    platform[idx], num_classes=self.num_platforms
                ).to(dtype=mu.dtype, device=mu.device)
                state = decayed_state + event_update.unsqueeze(1).expand_as(state)
                last_time = current_t
                has_history = torch.ones((), dtype=torch.bool, device=mu.device)

            if update_state:
                self.excitation_state = state.detach().cpu()
                self.last_time = last_time.detach().cpu()

            return {
                "event_index": order.detach().cpu(),
                "t": t_events[order].detach().cpu(),
                "platform": platform.detach().cpu(),
                "event_intensity": torch.stack(event_intensities).detach().cpu(),
                "total_intensity": torch.stack(total_intensities).detach().cpu(),
                "compensator": torch.stack(compensators).detach().cpu(),
                "event_nll": torch.stack(event_nlls).detach().cpu(),
                "excitation_mass": torch.stack(excitation_masses).detach().cpu(),
                "rolling_z": torch.stack(rolling_z).detach().cpu(),
                "alert_level": torch.stack(alert_levels).detach().cpu(),
            }

    def _rolling_z(self, value):
        if len(self._recent_nll) < 2:
            return 0.0
        sample = torch.tensor(list(self._recent_nll), dtype=torch.float)
        std = sample.std(unbiased=False).clamp_min(1e-6)
        return float((value - sample.mean()).item() / std.item())
