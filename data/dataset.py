import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import TemporalData

NUM_PLATFORMS = 3  # Reddit, HackerNews, GDELT

def load_simcity_temporal_data(pkl_path, base_msg_dim=16):
    df = pd.read_pickle(pkl_path)
    df = df.sort_values(by='t').reset_index(drop=True)

    src = torch.tensor(df['src'].values, dtype=torch.long)
    dst = torch.tensor(df['dst'].values, dtype=torch.long)
    t   = torch.tensor(df['t'].values,   dtype=torch.float)

    # Base message features
    msg_base = torch.tensor(df['msg'].tolist(), dtype=torch.float)
    assert msg_base.dim() == 2, \
        f"msg must be 2D, got {msg_base.shape}. Check for ragged rows."

    # Concatenate platform one-hot and GDELT volume into msg
    platform_idx    = torch.tensor(df['platform'].values, dtype=torch.long)
    platform_onehot = F.one_hot(platform_idx, num_classes=NUM_PLATFORMS).float()
    gdelt_col       = torch.tensor(
        df['gdelt_volume'].values, dtype=torch.float).unsqueeze(1)
    msg = torch.cat([msg_base, platform_onehot, gdelt_col], dim=1)

    # Virality regression target: log engagement count.
    # NaN for synthetic data — mask in loss computation.
    if 'log_engagement' in df.columns:
        y = torch.tensor(df['log_engagement'].values, dtype=torch.float)
    else:
        y = torch.full((len(df),), float('nan'))

    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y, platform=platform_idx)

    # --- Chronological split by TIME, not event count ---
    t_min, t_max = float(t.min()), float(t.max())
    t_range = t_max - t_min
    train_end_time = t_min + 0.70 * t_range
    val_end_time   = t_min + 0.80 * t_range

    train_idx = torch.where(t <  train_end_time)[0]
    val_idx   = torch.where((t >= train_end_time) & (t < val_end_time))[0]
    test_idx  = torch.where(t >= val_end_time)[0]

    # CRITICAL: TGN memory states s_v(t) and Hawkes history H_t MUST be
    # reset via model.reset_memory() at val and test boundaries in the
    # training loop. Failure to do so causes temporal leakage.

    return data, train_idx, val_idx, test_idx