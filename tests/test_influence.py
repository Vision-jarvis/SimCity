import torch
from models.influence import DynamicInfluenceScorer

def test_influence_scorer_initialization():
    scorer = DynamicInfluenceScorer(num_nodes=10)
    assert scorer.temporal_degree.sum() == 0
    assert scorer.engagement_velocity.sum() == 0
    assert torch.allclose(scorer.tpr.sum(), torch.tensor(1.0))
    
def test_influence_scorer_decay():
    scorer = DynamicInfluenceScorer(num_nodes=2, half_life_seconds=3600.0)
    # Give initial state
    scorer.temporal_degree[0] = 10.0
    scorer.engagement_velocity[0] = 5.0
    
    # 1 hour decay
    dt = torch.tensor(3600.0)
    decay = scorer.compute_decay(dt)
    assert torch.allclose(decay, torch.tensor(0.5))
    
    # Check manual decay function
    inf = scorer.get_current_influence(torch.tensor([0]), torch.tensor([3600.0]))
    
    weights = torch.softmax(scorer.omega, dim=0)
    expected_tpr = (1.0 / 2) * 0.5
    expected_td = 10.0 * 0.5
    expected_vel = 5.0 * 0.5
    
    expected_inf = weights[0] * expected_tpr + weights[1] * expected_vel + weights[2] * expected_td
    assert torch.allclose(inf[0], expected_inf)

def test_influence_scorer_forward():
    scorer = DynamicInfluenceScorer(num_nodes=3, half_life_seconds=3600.0, tpr_alpha=0.5)
    
    # Node 0 interacts with Node 1 at t=0
    src = torch.tensor([0])
    dst = torch.tensor([1])
    t = torch.tensor([0.0])
    
    inf_src, inf_dst = scorer(src, dst, t)
    
    # State should update
    assert scorer.temporal_degree[0] == 1.0
    assert scorer.temporal_degree[1] == 1.0
    assert scorer.temporal_degree[2] == 0.0
    
    # TPR transfer: 0 -> 1
    # Initial TPR was 1/3 = 0.333
    # Transfer = 0.5 * 0.333 = 0.166
    # TPR[1] = 0.333 + 0.166 = 0.5
    assert torch.allclose(scorer.tpr[1], torch.tensor(0.5))
    
    # Now node 1 interacts with node 2 at t=3600
    src2 = torch.tensor([1])
    dst2 = torch.tensor([2])
    t2 = torch.tensor([3600.0])
    
    inf_src2, inf_dst2 = scorer(src2, dst2, t2)
    
    # Node 1 td was 1.0, decasy to 0.5, then adds 1.0 -> 1.5
    assert torch.allclose(scorer.temporal_degree[1], torch.tensor(1.5))
    
    # Node 2 td was 0.0, gets 1.0
    assert torch.allclose(scorer.temporal_degree[2], torch.tensor(1.0))
    
    # TPR transfer 1 -> 2
    # Node 1 TPR was 0.5, decays to 0.25
    # Transfer = 0.5 * 0.25 = 0.125
    # Node 2 TPR was 0.333, decays to 0.166, adds 0.125 -> 0.2916
    assert torch.allclose(scorer.tpr[2], torch.tensor(1/3 * 0.5 + 0.125))
