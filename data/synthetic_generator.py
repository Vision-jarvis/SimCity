import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta

BASE_MSG_DIM = 128
NUM_PLATFORMS = 3       # 0: Reddit, 1: Hacker News, 2: GDELT/News
# Effective msg dim after dataset.py concatenation: BASE_MSG_DIM + NUM_PLATFORMS + 1 = 132
EFFECTIVE_MSG_DIM = BASE_MSG_DIM + NUM_PLATFORMS + 1


def compute_future_engagement(df, prediction_window_seconds=86400):
    """
    For each event at time t, computes how much engagement the same narrative
    receives in the window (t, t + prediction_window_seconds].
    """
    df = df.sort_values('t').reset_index(drop=True)
    t_vals = df['t'].values
    dst_vals = df['dst'].values
    n = len(df)
    future_counts = np.zeros(n, dtype=np.float32)

    for narrative_id in np.unique(dst_vals):
        idx = np.where(dst_vals == narrative_id)[0]
        t_narrative = t_vals[idx]

        t_end = t_narrative + prediction_window_seconds
        
        future_start = np.searchsorted(t_narrative, t_narrative, side='right')
        future_end   = np.searchsorted(t_narrative, t_end, side='right')
        
        future_counts[idx] = (future_end - future_start).astype(np.float32)

    return np.log1p(future_counts)

def generate_synthetic_multiplex_data(
    num_events=10000,
    num_users=1000,
    num_narratives=100,
    msg_dim=BASE_MSG_DIM,
    seed=42
):
    np.random.seed(seed)
    print(f"Generating {num_events} synthetic events (seed={seed})...")

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    total_seconds = int((end_date - start_date).total_seconds())

    timestamps_sec = np.sort(
        np.random.randint(0, total_seconds, size=num_events)
    ).astype(float)

    user_weights = np.random.pareto(a=2.0, size=num_users) + 1
    user_probs = user_weights / user_weights.sum()

    narrative_weights = np.random.pareto(a=1.5, size=num_narratives) + 1
    narrative_probs = narrative_weights / narrative_weights.sum()

    src_ids = np.random.choice(num_users, size=num_events, p=user_probs)
    dst_ids = np.random.choice(num_narratives, size=num_events, p=narrative_probs)
    dst_ids += num_users

    platforms = np.random.choice(
        [0, 1, 2], size=num_events, p=[0.60, 0.25, 0.15]
    )

    num_bins = int(total_seconds / (15 * 60)) + 1
    gdelt_volume = np.zeros(num_bins)
    gdelt_volume[0] = 10.0
    for b in range(1, num_bins):
        gdelt_volume[b] = max(
            0.0, 0.85 * gdelt_volume[b - 1] + np.random.normal(0, 2)
        )
    spike_bins = np.random.choice(num_bins, size=5, replace=False)
    gdelt_volume[spike_bins] += np.random.uniform(30, 80, size=5)

    bin_indices = (timestamps_sec // (15 * 60)).astype(int)
    gdelt_features = gdelt_volume[bin_indices]

    base_embeddings = np.random.randn(num_narratives, msg_dim)
    msg_embeddings = np.zeros((num_events, msg_dim))
    for i in range(num_events):
        narrative_idx = dst_ids[i] - num_users
        msg_embeddings[i] = (
            base_embeddings[narrative_idx]
            + np.random.normal(0, 0.1, size=msg_dim)
        )

    df = pd.DataFrame({
        'src': src_ids,
        'dst': dst_ids,
        't': timestamps_sec,
        'platform': platforms,
        'gdelt_volume': gdelt_features,
    })
    df['msg'] = list(msg_embeddings)

    cascade_rows = []
    for _, row in df[df['platform'] == 0].iterrows():
        if np.random.rand() < 0.30:
            delay = np.random.exponential(scale=3600)
            new_t = row['t'] + delay
            if new_t < total_seconds:
                narrative_idx = int(row['dst']) - num_users
                cascade_rows.append({
                    'src': np.random.choice(num_users, p=user_probs),
                    'dst': row['dst'],
                    't': new_t,
                    'platform': 1,
                    'gdelt_volume': gdelt_volume[int(new_t) // (15 * 60)],
                    'msg': (base_embeddings[narrative_idx]
                            + np.random.normal(0, 0.1, size=msg_dim)),
                })

    if cascade_rows:
        cascade_df = pd.DataFrame(cascade_rows)
        cascade_df['msg'] = list(cascade_df['msg'])
        df = pd.concat([df, cascade_df], ignore_index=True)

    df = df.sort_values('t').reset_index(drop=True)
    df['log_engagement'] = compute_future_engagement(df, prediction_window_seconds=86400)
    
    print(f"  Total events after cascade injection: {len(df)}")
    print(f"  Effective msg dim (after dataset.py concat): {EFFECTIVE_MSG_DIM}")
    print(f"  Platform distribution:\n{df['platform'].value_counts().sort_index()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=int, default=10000)
    parser.add_argument('--out', type=str, default='synthetic_events.pkl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_multiplex_data(
        num_events=args.events, seed=args.seed
    )
    df.to_pickle(args.out)
    print(f"Saved to {args.out}")