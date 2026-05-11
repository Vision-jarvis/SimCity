import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from data.dataset import load_simcity_temporal_data
from models.tgn_core import SimCityTGN
from models.virality_head import PlatformPooling, ViralityHead
from models.loss import SimCityLoss
from models.hawkes import StreamingHawkesLoss
from evaluate import evaluate_model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print("Loading synthetic dataset...")
    data, train_idx, val_idx, test_idx = load_simcity_temporal_data('data/synthetic_events.pkl')
    train_data = data[train_idx]
    val_data = data[val_idx]
    test_data = data[test_idx]
    
    # Temporal Data Loaders
    train_loader = TemporalDataLoader(train_data, batch_size=200)
    val_loader = TemporalDataLoader(val_data, batch_size=200)
    test_loader = TemporalDataLoader(test_data, batch_size=200)
    
    # Temporal Neighborhood Loader (for GraphAttentionEmbedding)
    neighbor_loader = LastNeighborLoader(train_data.num_nodes, size=10, device=device)
    
    # 2. Instantiate Models
    num_nodes = train_data.num_nodes
    num_platforms = 3
    embedding_dim = 128
    raw_msg_dim = train_data.msg.size(-1)
    time_dim = 128
    
    tgn = SimCityTGN(num_nodes, raw_msg_dim, embedding_dim, time_dim, embedding_dim).to(device)
    pooling = PlatformPooling(embedding_dim, num_platforms).to(device)
    virality_head = ViralityHead(embedding_dim, num_platforms).to(device)
    loss_module = SimCityLoss().to(device)
    hawkes_loss_fn = StreamingHawkesLoss(num_platforms=num_platforms).to(device)
    
    valid_y = train_data.y[~torch.isnan(train_data.y)]
    if valid_y.numel() > 0:
        virality_head.target_mean.fill_(valid_y.mean().item())
        virality_head.target_std.fill_(valid_y.std(unbiased=False).clamp_min(1e-6).item())
    
    optimizer = Adam([
        {'params': tgn.parameters()},
        {'params': pooling.parameters()},
        {'params': virality_head.parameters()},
        {'params': loss_module.parameters(), 'lr': 1e-2} # Kendall tau log_vars are sensitive
    ], lr=1e-3)
    
    # Simple negative sampler for InfoNCE
    def sample_negatives(dst, num_nodes, num_negatives=20):
        # Shape: (B, M)
        return torch.randint(0, num_nodes, (dst.size(0), num_negatives), device=dst.device)
    
    epochs = 5
    # Before training begins — once
    tgn.reset_memory()
    neighbor_loader.reset_state()
    hawkes_loss_fn.reset_state()
    
    for epoch in range(epochs):
        tgn.train()
        pooling.train()
        virality_head.train()
        loss_module.train()
        hawkes_loss_fn.reset_state()
        
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            
            # Sample temporal neighborhood
            n_id, edge_index, e_id = neighbor_loader(torch.cat([src, pos_dst]))
            
            # Update neighborhood loader (for future batches)
            neighbor_loader.insert(src, pos_dst)
            
            # 1. TGN Forward Pass
            # Pass msg directly as mock edge_attr for the synthetic run
            # In a real run, this aligns precisely with e_id
            edge_attr = msg[e_id] if e_id is not None and e_id.numel() > 0 and e_id.max() < msg.size(0) else None
            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.size(1), raw_msg_dim), device=device)
            
            # TGN embeddings for the nodes in the computational graph
            h = tgn(n_id, edge_index, edge_attr, src, pos_dst, t, msg)
            
            # Extract src and dst embeddings
            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            h_src = h[assoc[src]]
            h_dst_pos = h[assoc[pos_dst]]
            
            # 2. InfoNCE Negative Sampling
            neg_dst = sample_negatives(pos_dst, num_nodes, num_negatives=20)
            neg_dst_flat = neg_dst.view(-1)
            z_neg, _ = tgn.memory(neg_dst_flat)
            h_dst_neg = tgn.embedding_mlp(z_neg).view(src.size(0), 20, -1)
            
            # 3. Platform Pooling
            platform_ids = batch.platform
            # Influence scores: simplified I(v,t) using embedding norm as proxy
            influence_scores = h_src.norm(dim=-1).detach()
            h_P = pooling(h_src, platform_ids, influence_scores)
            
            # 4. Virality Head Forward Pass
            gdelt_volume = msg[:, -1]
            beta, mu, alpha, gamma, pred_virality = virality_head(
                h_src, h_dst_pos, h_P, gdelt_volume
            )
            
            # 5. Loss Calculation
            l_tgn = loss_module.compute_tgn_loss(h_src, h_dst_pos, h_dst_neg)
            l_virality = loss_module.compute_virality_loss(pred_virality, batch.y)
            
            l_hawkes = hawkes_loss_fn(t, platform_ids, mu, alpha, gamma)
            
            loss = loss_module(l_tgn, l_hawkes, l_virality)
            
            loss.backward()
            optimizer.step()
            
            # Detach memory to prevent backpropagating through time forever
            tgn.memory.detach()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")
        
        # Validation
        print(f"Epoch {epoch+1} | Running Validation...")
        # Before validation — reset
        tgn.reset_memory()
        neighbor_loader.reset_state()
        hawkes_loss_fn.reset_state()
        evaluate_model(
            tgn, pooling, virality_head, val_loader, neighbor_loader,
            num_nodes, num_platforms, raw_msg_dim, device, hawkes_loss_fn
        )
        
        # Before resuming training next epoch — reset again to clean val contamination
        tgn.reset_memory()
        neighbor_loader.reset_state()
        hawkes_loss_fn.reset_state()

    print("\n--- Running Sanity Checks ---")
    
    # 1. Naive mean baseline
    train_mean = train_data.y[~train_data.y.isnan()].mean().item()
    naive_preds = torch.full_like(val_data.y, train_mean)
    mask = ~val_data.y.isnan()
    naive_mae = (naive_preds[mask] - val_data.y[mask]).abs().mean().item()
    print(f"Naive mean baseline MAE: {naive_mae:.4f}")
    
    # 2. Shuffled-timestamp generalization check
    import numpy as np
    import copy
    
    shuffled_val_data = copy.deepcopy(val_data)
    shuffled_t = np.random.permutation(val_data.t.cpu().numpy())
    shuffled_val_data.t = torch.tensor(shuffled_t, device=val_data.t.device)
    
    shuffled_val_loader = TemporalDataLoader(shuffled_val_data, batch_size=200, neg_sampling_ratio=0.0)
    
    print("Running Shuffled Validation...")
    tgn.reset_memory()
    neighbor_loader.reset_state()
    hawkes_loss_fn.reset_state()
    evaluate_model(
        tgn, pooling, virality_head, shuffled_val_loader, neighbor_loader,
        num_nodes, num_platforms, raw_msg_dim, device, hawkes_loss_fn
    )
    print("--- End of Sanity Checks ---")

if __name__ == "__main__":
    train()
