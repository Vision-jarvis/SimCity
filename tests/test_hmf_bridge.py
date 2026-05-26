import torch
import pytest
from models.hmf_bridge import DegreeCorrelatedHMF

def test_hmf_bridge_degree_update():
    bridge = DegreeCorrelatedHMF(max_degree_bin=5)
    
    # Edges between:
    # 2 -> 3
    # 2 -> 3
    # 4 -> 1
    
    k_u = torch.tensor([2, 2, 4], dtype=torch.float32)
    k_v = torch.tensor([3, 3, 1], dtype=torch.float32)
    
    bridge.update_degree_distribution(k_u, k_v)
    
    # Expected counts:
    # (2,3) = 2, (3,2) = 2
    # (4,1) = 1, (1,4) = 1
    assert bridge.edge_counts[2, 3] == 2
    assert bridge.edge_counts[3, 2] == 2
    assert bridge.edge_counts[4, 1] == 1
    assert bridge.edge_counts[1, 4] == 1
    
    # Expected probabilities P(k' | k):
    # For k=2, only connected to 3, so P(3|2) = 1.0
    assert torch.allclose(bridge.P_k_given_k[2, 3], torch.tensor(1.0))
    # For k=3, only connected to 2, so P(2|3) = 1.0
    assert torch.allclose(bridge.P_k_given_k[3, 2], torch.tensor(1.0))
    # For k=4, only connected to 1, so P(1|4) = 1.0
    assert torch.allclose(bridge.P_k_given_k[4, 1], torch.tensor(1.0))

def test_hmf_bridge_forward():
    bridge = DegreeCorrelatedHMF(max_degree_bin=5)
    
    # Fake some edges to set up P(k' | k)
    k_u = torch.tensor([1, 1], dtype=torch.float32)
    k_v = torch.tensor([2, 3], dtype=torch.float32)
    bridge.update_degree_distribution(k_u, k_v)
    
    # P(2|1) = 0.5, P(3|1) = 0.5
    # P(1|2) = 1.0
    # P(1|3) = 1.0
    
    beta_hat = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    # k_v for these active nodes
    k_v_active = torch.tensor([1, 1, 2], dtype=torch.float32)
    
    # E[beta_hat] for k=1 is (0.1 + 0.2) / 2 = 0.15
    # E[beta_hat] for k=2 is 0.3
    # E[beta_hat] for k=3 is 0.0
    
    beta_macro = bridge(beta_hat, k_v_active)
    
    # Expected calculation:
    # Inner sum for k=1: P(2|1)*2*E[beta_2] + P(3|1)*3*E[beta_3]
    #                  = 0.5 * 2 * 0.3 + 0.5 * 3 * 0.0 = 0.3
    # Inner sum for k=2: P(1|2)*1*E[beta_1] = 1.0 * 1 * 0.15 = 0.15
    # Inner sum for k=3: P(1|3)*1*E[beta_1] = 1.0 * 1 * 0.15 = 0.15
    # Inner sum for k=0, 4, 5: 0 (since they have no incoming edges, P is identity, and E[beta] = 0)
    
    # beta_macro = mean(inner_sum) = (0.3 + 0.15 + 0.15 + 0 + 0 + 0) / 6 = 0.6 / 6 = 0.1
    
    assert torch.allclose(beta_macro, torch.tensor(0.1))
