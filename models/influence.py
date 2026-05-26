import torch
import torch.nn as nn

class DynamicInfluenceScorer(nn.Module):
    """
    Computes the Dynamic Influence Score I(v,t) from Section 6 of the SimCity architecture.
    I(v,t) = w_1 * TPR(v,t) + w_2 * dE(v,t)/dt + w_3 * C_D(v,t)
    Maintains O(1) state updates per event using exponential decay.
    """
    def __init__(self, num_nodes, half_life_seconds=3600.0, tpr_alpha=0.85):
        super().__init__()
        self.num_nodes = num_nodes
        self.half_life = float(half_life_seconds)
        self.decay_lambda = 0.69314718 / self.half_life  # ln(2) / half_life
        self.tpr_alpha = tpr_alpha
        
        # Learnable weights for the three components
        self.omega = nn.Parameter(torch.ones(3) / 3.0)
        
        # O(1) Stateful buffers
        self.register_buffer('last_update_time', torch.zeros(num_nodes, dtype=torch.float32))
        self.register_buffer('temporal_degree', torch.zeros(num_nodes, dtype=torch.float32))
        self.register_buffer('engagement_velocity', torch.zeros(num_nodes, dtype=torch.float32))
        self.register_buffer('tpr', torch.ones(num_nodes, dtype=torch.float32) / num_nodes)

    def reset_state(self):
        """Resets the state tracking for a new cascade window."""
        self.last_update_time.zero_()
        self.temporal_degree.zero_()
        self.engagement_velocity.zero_()
        self.tpr.fill_(1.0 / self.num_nodes)

    def compute_decay(self, dt):
        return torch.exp(-self.decay_lambda * dt.clamp_min(0.0))

    def forward(self, src, dst, t, update_state=True):
        """
        Calculates influence for the nodes involved in the current batch of events.
        Events must be sorted chronologically within the batch.
        
        Args:
            src: (B,) source nodes
            dst: (B,) destination nodes
            t: (B,) event timestamps in seconds
            update_state: if True, updates the internal buffers
        
        Returns:
            influence_src: (B,) influence score for source nodes at time t
            influence_dst: (B,) influence score for destination nodes at time t
        """
        B = src.size(0)
        device = src.device
        
        # We process sequentially in the batch to maintain O(1) causality.
        # For pure PyTorch optimization without a loop, we could scatter/gather,
        # but causal streaming requires iterative or semi-iterative updates.
        # Since this is a lightweight PyTorch proxy, we'll do a fast loop.
        inf_src = torch.zeros(B, device=device, dtype=torch.float32)
        inf_dst = torch.zeros(B, device=device, dtype=torch.float32)
        
        state_td = self.temporal_degree.clone()
        state_vel = self.engagement_velocity.clone()
        state_tpr = self.tpr.clone()
        state_lut = self.last_update_time.clone()
        
        weights = torch.softmax(self.omega, dim=0)
        
        for i in range(B):
            u = src[i]
            v = dst[i]
            current_t = t[i]
            
            # 1. Decay state for u and v up to current_t
            dt_u = current_t - state_lut[u]
            dt_v = current_t - state_lut[v]
            
            decay_u = self.compute_decay(dt_u)
            decay_v = self.compute_decay(dt_v)
            
            # Apply decay
            state_td[u] *= decay_u
            state_td[v] *= decay_v
            
            state_vel[u] *= decay_u
            state_vel[v] *= decay_v
            
            state_tpr[u] *= decay_u
            state_tpr[v] *= decay_v
            
            # Compute current influence *before* the event
            inf_src[i] = weights[0] * state_tpr[u] + weights[1] * state_vel[u] + weights[2] * state_td[u]
            inf_dst[i] = weights[0] * state_tpr[v] + weights[1] * state_vel[v] + weights[2] * state_td[v]
            
            # 2. Update state with the new event
            state_td[u] += 1.0
            state_td[v] += 1.0
            
            # Velocity: lambda increment (instantaneous rate)
            state_vel[u] += self.decay_lambda
            state_vel[v] += self.decay_lambda
            
            # TPR: Random walk step from u to v
            # TPR flows from source to destination
            transfer = self.tpr_alpha * state_tpr[u]
            state_tpr[v] += transfer
            
            state_lut[u] = current_t
            state_lut[v] = current_t
            
        if update_state:
            self.temporal_degree = state_td.detach()
            self.engagement_velocity = state_vel.detach()
            self.tpr = state_tpr.detach()
            self.last_update_time = state_lut.detach()
            
        return inf_src, inf_dst

    def get_current_influence(self, nodes, current_t):
        """Returns the decayed influence score for specific nodes at current_t."""
        dt = current_t - self.last_update_time[nodes]
        decay = self.compute_decay(dt)
        
        td = self.temporal_degree[nodes] * decay
        vel = self.engagement_velocity[nodes] * decay
        tpr = self.tpr[nodes] * decay
        
        weights = torch.softmax(self.omega, dim=0)
        return weights[0] * tpr + weights[1] * vel + weights[2] * td
