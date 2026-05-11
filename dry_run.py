# dry_run.py
import torch
from data.dataset import load_simcity_temporal_data
from models.tgn_core import SimCityTGN
from models.virality_head import PlatformPooling, ViralityHead
from models.loss import SimCityLoss
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

device = torch.device('cpu')  # CPU only for dry-run — easier stack traces

data, train_idx, val_idx, test_idx = load_simcity_temporal_data('data/synthetic_events.pkl')
train_data = data[train_idx]
train_loader = TemporalDataLoader(train_data, batch_size=50)  # small batch
neighbor_loader = LastNeighborLoader(train_data.num_nodes, size=10, device=device)

num_nodes = train_data.num_nodes
num_platforms = 3
embedding_dim = 64  # smaller than production for speed
raw_msg_dim = train_data.msg.size(-1)
time_dim = 64

tgn = SimCityTGN(num_nodes, raw_msg_dim, embedding_dim, time_dim, embedding_dim).to(device)
pooling = PlatformPooling(embedding_dim, num_platforms).to(device)
virality_head = ViralityHead(embedding_dim, num_platforms).to(device)
loss_module = SimCityLoss().to(device)

tgn.reset_memory()
neighbor_loader.reset_state()

batch = next(iter(train_loader))
batch = batch.to(device)
src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

print(f"[DATA] batch size: {src.size(0)}")
print(f"[DATA] msg shape: {msg.shape}")
print(f"[DATA] t dtype: {t.dtype}")
print(f"[DATA] platform unique values: {batch.platform.unique()}")
print(f"[DATA] y (virality targets) sample: {batch.y[:5]}")
print(f"[DATA] NaN count in y: {batch.y.isnan().sum().item()}")

n_id, edge_index, e_id = neighbor_loader(torch.cat([src, pos_dst]))
neighbor_loader.insert(src, pos_dst)

print(f"[NEIGHBOR] n_id shape: {n_id.shape}")
print(f"[NEIGHBOR] edge_index shape: {edge_index.shape}")

edge_attr = msg[e_id] if (e_id is not None and e_id.numel() > 0 
                          and e_id.max() < msg.size(0)) else \
            torch.zeros(edge_index.size(1), raw_msg_dim, device=device)

h = tgn(n_id, edge_index, edge_attr, src, pos_dst, t, msg)
print(f"[TGN] h shape: {h.shape}")
assert h.shape == (n_id.size(0), embedding_dim), \
    f"Expected ({n_id.size(0)}, {embedding_dim}), got {h.shape}"

assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
assoc[n_id] = torch.arange(n_id.size(0), device=device)
h_src = h[assoc[src]]
h_dst_pos = h[assoc[pos_dst]]
print(f"[TGN] h_src shape: {h_src.shape}, h_dst_pos shape: {h_dst_pos.shape}")

neg_dst = torch.randint(0, num_nodes, (src.size(0), 20), device=device)
z_neg, _ = tgn.memory(neg_dst.view(-1))
h_dst_neg = tgn.embedding_mlp(z_neg).view(src.size(0), 20, embedding_dim)
print(f"[TGN] h_dst_neg shape: {h_dst_neg.shape}")
assert h_dst_neg.shape == (src.size(0), 20, embedding_dim), \
    f"Negative embedding shape wrong: {h_dst_neg.shape}"

platform_ids = getattr(batch, 'platform', torch.randint(0, num_platforms, (n_id.size(0),), device=device))
# Wait, the prompt says platform_ids = batch.platform but batch.platform length is the number of events, not n_id.
# If batch.platform doesn't align with n_id, it might fail. The user's prompt says:
# platform_ids = batch.platform
# Let's use exactly what the user provided.
platform_ids = batch.platform
influence_scores = h_src.norm(dim=-1).detach()
h_P = pooling(h_src, platform_ids, influence_scores)
print(f"[POOLING] h_P shape: {h_P.shape}")
assert h_P.shape == (num_platforms, embedding_dim), \
    f"Expected ({num_platforms}, {embedding_dim}), got {h_P.shape}"

beta, alpha, gamma, pred_virality = virality_head(h_src, h_dst_pos, h_P)
print(f"[VIRALITY HEAD] beta shape: {beta.shape}")
print(f"[VIRALITY HEAD] alpha shape: {alpha.shape}")
print(f"[VIRALITY HEAD] gamma shape: {gamma.shape}")
print(f"[VIRALITY HEAD] pred_virality shape: {pred_virality.shape}")
assert alpha.shape[-2:] == (num_platforms, num_platforms), \
    f"alpha must be P×P, got {alpha.shape}"
assert gamma.shape[-2:] == (num_platforms, num_platforms), \
    f"gamma must be P×P, got {gamma.shape}"
assert (alpha >= 0).all(), "alpha has negative values — Softplus failed"
assert (gamma > 0).all(), "gamma has non-positive values — epsilon missing"

l_tgn = loss_module.compute_tgn_loss(h_src, h_dst_pos, h_dst_neg)
l_virality = loss_module.compute_virality_loss(pred_virality, batch.y)
l_hawkes = torch.tensor(0.0, device=device)
print(f"[LOSS] l_tgn: {l_tgn.item():.4f}")
print(f"[LOSS] l_virality: {l_virality.item():.4f}")

loss = loss_module(l_tgn, l_hawkes, l_virality)
print(f"[LOSS] combined loss: {loss.item():.4f}")
assert not torch.isnan(loss), "Combined loss is NaN"
assert not torch.isinf(loss), "Combined loss is Inf"

loss.backward()
print("[BACKWARD] Gradient pass successful")

for name, param in list(tgn.named_parameters())[:5]:
    if param.grad is not None:
        print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
    else:
        print(f"  {name}: NO GRADIENT — likely detached or unused")

print("\n--- DRY RUN PASSED ---")
