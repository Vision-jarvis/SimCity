import torch
from models.deffuant import SmoothDeffuant

def test_smooth_deffuant_radicalization():
    model = SmoothDeffuant(alpha_r=0.5)
    
    total_edges = torch.tensor([10, 10, 0], dtype=torch.long)
    cross_edges = torch.tensor([5, 0, 0], dtype=torch.long)
    num_exposed = torch.tensor([100, 100, 0], dtype=torch.long)
    num_rejected = torch.tensor([25, 100, 0], dtype=torch.long)
    
    # Node 0: struct=1-5/10=0.5, hist=25/100=0.25 => rad = 0.5*0.5 + 0.5*0.25 = 0.375
    # Node 1: struct=1-0/10=1.0, hist=100/100=1.0 => rad = 0.5*1.0 + 0.5*1.0 = 1.0
    # Node 2: struct=1.0 (no edges), hist=0.0 (no exp) => rad = 0.5*1.0 + 0.0 = 0.5
    
    rad = model.compute_radicalization(total_edges, cross_edges, num_exposed, num_rejected)
    
    assert torch.allclose(rad[0], torch.tensor(0.375))
    assert torch.allclose(rad[1], torch.tensor(1.0))
    assert torch.allclose(rad[2], torch.tensor(0.5))

def test_smooth_deffuant_derivative():
    # epsilon_base = 0.5, kappa=100 => rho = 200
    model = SmoothDeffuant(eta=0.1, kappa=100.0, epsilon_base=0.5)
    
    x_v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    x_c = torch.tensor([0.2, 0.6, 0.5], dtype=torch.float32) # Within, outside, boundary
    eps_p = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    
    dx_dt = model.compute_derivative(x_v, x_c, eps_p)
    
    # Node 0: diff=0.2, eps=0.5 => z = 200 * 0.3 = 60 => sigma = 1.0
    # dx_dt = 0.1 * 0.2 * 1.0 = 0.02
    assert torch.allclose(dx_dt[0], torch.tensor(0.02), atol=1e-4)
    
    # Node 1: diff=0.6, eps=0.5 => z = 200 * (-0.1) = -20 => sigma = 0.0
    # dx_dt = 0.1 * 0.6 * 0.0 = 0.0
    assert torch.allclose(dx_dt[1], torch.tensor(0.0), atol=1e-4)
    
    # Node 2: diff=0.5, eps=0.5 => z = 0 => sigma = 0.5
    # dx_dt = 0.1 * 0.5 * 0.5 = 0.025
    assert torch.allclose(dx_dt[2], torch.tensor(0.025), atol=1e-4)

def test_smooth_deffuant_forward():
    model = SmoothDeffuant(eta=0.1)
    
    x_v = torch.tensor([0.0], dtype=torch.float32)
    x_c = torch.tensor([0.2], dtype=torch.float32)
    dt = 1.0
    
    total_edges = torch.tensor([10], dtype=torch.long)
    cross_edges = torch.tensor([10], dtype=torch.long)
    num_exposed = torch.tensor([10], dtype=torch.long)
    num_rejected = torch.tensor([0], dtype=torch.long)
    
    # rad = 0 (completely connected, no rejections)
    # eps_p = 0.5
    
    x_v_new, rejected, rad = model(x_v, x_c, dt, total_edges, cross_edges, num_exposed, num_rejected)
    
    assert torch.allclose(rad[0], torch.tensor(0.0))
    assert not rejected[0]
    assert x_v_new[0] > 0.0 # should move towards x_c
