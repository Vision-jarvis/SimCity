import torch
from data.dataset import load_simcity_temporal_data

data, train_idx, val_idx, test_idx = load_simcity_temporal_data('synthetic_events.pkl')
train_data = data[train_idx]
y = train_data.y
valid_y = y[~torch.isnan(y)]

print(f"New y mean: {valid_y.mean().item():.4f}")
print(f"New y std:  {valid_y.std().item():.4f}")
print(f"New y min:  {valid_y.min().item():.4f}")
print(f"New y max:  {valid_y.max().item():.4f}")

# Critical check: variance should now differ across events of the same narrative
sample_dst = train_data.dst[0].item()
same_narrative_mask = (train_data.dst == sample_dst)
print(f"\nTargets for narrative {sample_dst}:")
print(train_data.y[same_narrative_mask][:10])
