import torch
from data.dataset import load_simcity_temporal_data
from torch_geometric.loader import TemporalDataLoader
from models.tgn_core import SimCityTGN
from models.virality_head import ViralityHead
from models.loss import SimCityLoss
from models.hawkes import StreamingHawkesLoss
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, train_idx, val_idx, test_idx = load_simcity_temporal_data('data/synthetic_events.pkl')
train_data = data[train_idx]
val_data = data[val_idx]
num_nodes = max(data.src.max(), data.dst.max()) + 1
raw_msg_dim = data.msg.size(1)

print("\n--- PHASE 1: CHECK A & C (Original Weights) ---")
tgn = SimCityTGN(num_nodes=num_nodes, raw_msg_dim=raw_msg_dim, memory_dim=64, time_dim=64, embedding_dim=64).to(device)
tgn.reset_memory()
from models.virality_head import PlatformPooling
pooling = PlatformPooling(embedding_dim=64, num_platforms=3).to(device)
virality_head = ViralityHead(embedding_dim=64, num_platforms=3).to(device)
from torch_geometric.nn.models.tgn import LastNeighborLoader
neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
loss_module = SimCityLoss().to(device)
hawkes_loss_fn = StreamingHawkesLoss(num_platforms=3).to(device)
hawkes_loss_fn.reset_state()
optimizer = torch.optim.Adam(list(tgn.parameters()) + list(pooling.parameters()) + list(virality_head.parameters()) + list(loss_module.parameters()), lr=1e-3)
train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=0.0)
batch = next(iter(train_loader)).to(device)

optimizer.zero_grad()
src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
n_id, edge_index, e_id = neighbor_loader(torch.cat([src, pos_dst]))
neighbor_loader.insert(src, pos_dst)
edge_attr = msg[e_id] if e_id is not None and e_id.numel() > 0 and e_id.max() < msg.size(0) else None
if edge_attr is None:
    edge_attr = torch.zeros((edge_index.size(1), raw_msg_dim), device=device)

h = tgn(n_id, edge_index, edge_attr, src, pos_dst, t, msg)

assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
assoc[n_id] = torch.arange(n_id.size(0), device=device)
h_src = h[assoc[src]]
h_dst_pos = h[assoc[pos_dst]]

h_src.retain_grad()
h_dst_pos.retain_grad()

neg_dst = torch.randint(0, num_nodes, (src.size(0), 20), dtype=torch.long, device=device)
neg_dst_flat = neg_dst.view(-1)
z_neg, _ = tgn.memory(neg_dst_flat)
h_dst_neg = tgn.embedding_mlp(z_neg).view(src.size(0), 20, -1)

platform_ids = batch.platform
influence_scores = h_src.norm(dim=-1).detach()
h_P = pooling(h_src, platform_ids, influence_scores)

gdelt_volume = msg[:, -1]
beta, mu, alpha, gamma, pred_virality = virality_head(
    h_src, h_dst_pos, h_P, gdelt_volume
)

l_tgn = loss_module.compute_tgn_loss(h_src, h_dst_pos, h_dst_neg)
l_hawkes = hawkes_loss_fn(t, platform_ids, mu, alpha, gamma)
l_virality = loss_module.compute_virality_loss(pred_virality, batch.y)

loss = loss_module(l_tgn, l_hawkes, l_virality)

print(f"Check A - log_vars after init: {loss_module.log_vars.data}")
print(f"Check A - precision weights: {torch.exp(-loss_module.log_vars).data}")
print(
    f"Check A - l_tgn raw: {l_tgn.item():.4f}, "
    f"l_hawkes raw: {l_hawkes.item():.4f}, "
    f"l_virality raw: {l_virality.item():.4f}"
)

loss.backward()

print("\nCheck C - Gradient Flow:")
print(f"  h_src grad norm: {h_src.grad.norm().item() if h_src.grad is not None else 'NONE'}")
print(f"  h_dst_pos grad norm: {h_dst_pos.grad.norm().item() if h_dst_pos.grad is not None else 'NONE'}")
for name, param in virality_head.named_parameters():
    if param.grad is not None:
        print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
    else:
        print(f"  {name}: NO GRADIENT")

print("\n--- PHASE 2: CHECK B (Flat Equal Weights) ---")
# Let's train for 5 epochs with flat weights using the train.py loop basically.
import types
from evaluate import evaluate_model
def flat_forward(self, l_tgn, l_hawkes, l_virality):
    return l_tgn + l_hawkes + l_virality

loss_module.forward = types.MethodType(flat_forward, loss_module)

for epoch in range(1, 6):
    tgn.train()
    virality_head.train()
    tgn.reset_memory()
    neighbor_loader.reset_state()
    hawkes_loss_fn.reset_state()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        n_id, edge_index, e_id = neighbor_loader(torch.cat([src, pos_dst]))
        neighbor_loader.insert(src, pos_dst)
        edge_attr = msg[e_id] if e_id is not None and e_id.numel() > 0 and e_id.max() < msg.size(0) else None
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), raw_msg_dim), device=device)
        
        h = tgn(n_id, edge_index, edge_attr, src, pos_dst, t, msg)
        
        assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        h_src = h[assoc[src]]
        h_dst_pos = h[assoc[pos_dst]]
        
        neg_dst = torch.randint(0, num_nodes, (src.size(0), 20), dtype=torch.long, device=device)
        neg_dst_flat = neg_dst.view(-1)
        z_neg, _ = tgn.memory(neg_dst_flat)
        h_dst_neg = tgn.embedding_mlp(z_neg).view(src.size(0), 20, -1)
        
        platform_ids = batch.platform
        influence_scores = h_src.norm(dim=-1).detach()
        h_P = pooling(h_src, platform_ids, influence_scores)
        
        gdelt_volume = msg[:, -1]
        beta, mu, alpha, gamma, pred_virality = virality_head(
            h_src, h_dst_pos, h_P, gdelt_volume
        )
        l_tgn = loss_module.compute_tgn_loss(h_src, h_dst_pos, h_dst_neg)
        l_hawkes = hawkes_loss_fn(t, platform_ids, mu, alpha, gamma)
        l_virality = loss_module.compute_virality_loss(pred_virality, batch.y)
        loss = loss_module(l_tgn, l_hawkes, l_virality)
        
        loss.backward()
        optimizer.step()
        tgn.memory.detach()
        total_loss += loss.item()
        
    print(f"Epoch {epoch} | Train Loss (Flat): {total_loss / len(train_loader):.4f}")

tgn.reset_memory()
neighbor_loader.reset_state()
hawkes_loss_fn.reset_state()
val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=0.0)
eval_metrics = evaluate_model(
    tgn, pooling, virality_head, val_loader, neighbor_loader,
    num_nodes, 3, raw_msg_dim, device, hawkes_loss_fn
)
print(f"\nCheck B - Final Val MAE (Flat weights): {eval_metrics['virality_mae']:.4f}")
