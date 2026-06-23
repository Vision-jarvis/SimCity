import torch
from torch.optim import AdamW
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from data.dataset import load_simcity_temporal_data
from models.tgn_core import SimCityTGN
from models.virality_head import PlatformPooling, ViralityHead
from models.loss import SimCityLoss
from models.hawkes import StreamingHawkesLoss
from models.deffuant import SmoothDeffuant
from models.influence import DynamicInfluenceScorer
from models.hmf_bridge import DegreeCorrelatedHMF
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
    embedding_dim = 256
    raw_msg_dim = train_data.msg.size(-1)
    time_dim = 256
    
    tgn = SimCityTGN(num_nodes, raw_msg_dim, embedding_dim, time_dim, embedding_dim).to(device)
    pooling = PlatformPooling(embedding_dim, num_platforms).to(device)
    virality_head = ViralityHead(embedding_dim, num_platforms).to(device)
    loss_module = SimCityLoss().to(device)
    hawkes_loss_fn = StreamingHawkesLoss(num_platforms=num_platforms).to(device)
    
    deffuant = SmoothDeffuant().to(device)
    influence_scorer = DynamicInfluenceScorer(num_nodes=num_nodes).to(device)
    hmf_bridge = DegreeCorrelatedHMF().to(device)
    
    # State for Deffuant
    node_opinions = torch.rand(num_nodes, device=device)
    node_exposed = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_rejected = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    valid_y = train_data.y[~torch.isnan(train_data.y)]
    if valid_y.numel() > 0:
        virality_head.target_mean.fill_(valid_y.mean().item())
        virality_head.target_std.fill_(valid_y.std(unbiased=False).clamp_min(1e-6).item())
    
    optimizer = AdamW([
        {'params': tgn.parameters()},
        {'params': pooling.parameters()},
        {'params': virality_head.parameters()},
        {'params': loss_module.parameters(), 'lr': 1e-2} # Kendall tau log_vars are sensitive
    ], lr=1e-3, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Simple negative sampler for InfoNCE
    def sample_negatives(dst, num_nodes, num_negatives=50):
        # Shape: (B, M)
        return torch.randint(0, num_nodes, (dst.size(0), num_negatives), device=dst.device)
    
    epochs = 15
    # Before training begins — once
    tgn.reset_memory()
    neighbor_loader.reset_state()
    hawkes_loss_fn.reset_state()
    influence_scorer.reset_state()
    
    for epoch in range(epochs):
        tgn.train()
        pooling.train()
        virality_head.train()
        loss_module.train()
        deffuant.train()
        influence_scorer.train()
        hmf_bridge.train()
        hawkes_loss_fn.reset_state()
        influence_scorer.reset_state()
        
        # Reset deffuant states per epoch for synthetic stream
        node_opinions.random_()
        node_exposed.zero_()
        node_rejected.zero_()
        
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
            
            # Behavioral Update (Deffuant)
            # Mock content opinion using first feature of msg
            x_c = msg[:, 0]
            dt_step = torch.ones_like(src, dtype=torch.float32)
            
            x_v_src = node_opinions[src]
            
            x_v_new, rejected, rad = deffuant(
                x_v_src, x_c, dt_step, 
                influence_scorer.temporal_degree[src].long(),
                (influence_scorer.temporal_degree[src] * 0.5).long(), # Mock cross-community
                node_exposed[src],
                node_rejected[src]
            )
            
            # Update state
            node_opinions[src] = x_v_new
            node_exposed[src] += 1
            node_rejected[src] += rejected.long()
            
            # Full rad vector for TGN forward
            full_rad = torch.zeros(num_nodes, 1, device=device)
            full_rad[src, 0] = rad
            
            # TGN embeddings for the nodes in the computational graph
            h = tgn(n_id, edge_index, edge_attr, src, pos_dst, t, msg, full_rad)
            
            # Extract src and dst embeddings
            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            h_src = h[assoc[src]]
            h_dst_pos = h[assoc[pos_dst]]
            
            # 2. InfoNCE Negative Sampling
            neg_dst = sample_negatives(pos_dst, num_nodes, num_negatives=50)
            neg_dst_flat = neg_dst.view(-1)
            z_neg, _ = tgn.memory(neg_dst_flat)
            z_neg_with_rad = torch.cat([z_neg, full_rad[neg_dst_flat]], dim=-1)
            h_dst_neg = tgn.embedding_mlp(z_neg_with_rad).view(src.size(0), 50, -1)
            
            # 3. Platform Pooling
            platform_ids = batch.platform
            
            inf_src, inf_dst = influence_scorer(src, pos_dst, t)
            
            # Update HMF Bridge
            hmf_bridge.update_degree_distribution(influence_scorer.temporal_degree[src], influence_scorer.temporal_degree[pos_dst])
            
            h_P = pooling(h_src, platform_ids, inf_src.detach())
            
            # 4. Virality Head Forward Pass
            gdelt_volume = msg[:, -1]
            beta, mu, alpha, gamma, pred_virality = virality_head(
                h_src, h_dst_pos, h_P, gdelt_volume
            )
            
            # Exercise the HMF bridge forward pass (macroscopic beta is used by
            # the simulation engine; not part of the training loss here).
            hmf_bridge(beta, influence_scorer.temporal_degree[src])
            
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
        influence_scorer.reset_state()
        metrics = evaluate_model(
            tgn, pooling, virality_head, deffuant, influence_scorer, hmf_bridge, val_loader, neighbor_loader,
            num_nodes, num_platforms, raw_msg_dim, device, hawkes_loss_fn
        )
        
        # Step the scheduler based on validation metric (e.g. sum of MAE and Hawkes NLL if available)
        val_metric = metrics.get("virality_mae", float('inf')) + metrics.get("hawkes_nll", 0.0)
        scheduler.step(val_metric)
        
        # Before resuming training next epoch — reset again to clean val contamination
        tgn.reset_memory()
        neighbor_loader.reset_state()
        hawkes_loss_fn.reset_state()
        influence_scorer.reset_state()

    # Final held-out test evaluation (chronologically latest split).
    print("\n--- Final Test Set Evaluation ---")
    tgn.reset_memory()
    neighbor_loader.reset_state()
    hawkes_loss_fn.reset_state()
    influence_scorer.reset_state()
    test_metrics = evaluate_model(
        tgn, pooling, virality_head, deffuant, influence_scorer, hmf_bridge, test_loader, neighbor_loader,
        num_nodes, num_platforms, raw_msg_dim, device, hawkes_loss_fn
    )
    print(f"Test metrics: {test_metrics}")

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
    influence_scorer.reset_state()
    evaluate_model(
        tgn, pooling, virality_head, deffuant, influence_scorer, hmf_bridge, shuffled_val_loader, neighbor_loader,
        num_nodes, num_platforms, raw_msg_dim, device, hawkes_loss_fn
    )
    print("--- End of Sanity Checks ---")

if __name__ == "__main__":
    train()
