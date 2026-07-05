import numpy as np
import pandas as pd
import argparse
from datetime import datetime

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


# ---------------------------------------------------------------------------
# Ground-truth generative model.
#
# Event *times and platforms* are produced by a per-narrative, cross-platform
# Hawkes branching (immigrant-offspring) process. Each narrative n has its own
# P x P branching matrix A_n whose off-diagonal mass (Reddit -> HN -> News)
# is controlled by a latent "bridge" scalar b_n, and its own base volume r_n.
#
# This is deliberate: the future-engagement target then depends on the *recent
# excitation trajectory* (a temporal quantity), so a model that tracks memory
# and cross-platform excitation state (SimCity) can beat a static snapshot GNN
# (which only sees a narrative's average behaviour) and a single global-alpha
# Hawkes (which cannot represent per-narrative excitation). The latent
# (r_n, b_n) is embedded in the narrative message vector so it is recoverable.
# ---------------------------------------------------------------------------

# Mean offspring delay per platform (seconds). Reddit fast, News slower.
OFFSPRING_TAU = np.array([1.5 * 3600, 3.0 * 3600, 6.0 * 3600])


def _branching_matrix(bridge):
    """P x P matrix; A[i, j] = expected # children on platform i from a parent
    on platform j. Diagonal = self-excitation; the Reddit->HN->News chain grows
    with ``bridge``. Column sums are held < 0.85 for subcriticality/stability."""
    A = np.zeros((NUM_PLATFORMS, NUM_PLATFORMS))
    # Self-excitation (intra-platform). High => offspring-dominated (near-critical)
    # cascades, so future engagement is driven by the *recent excitation state*
    # rather than a narrative's static immigrant volume.
    np.fill_diagonal(A, 0.55)
    # Cross-platform transfer chain, scaled by the narrative's bridge factor
    A[1, 0] = 0.55 * bridge   # Reddit -> HN
    A[2, 1] = 0.55 * bridge   # HN -> News
    A[2, 0] = 0.20 * bridge   # Reddit -> News (weak direct)
    # Clamp column sums near (but below) criticality for a strong, stable
    # temporal signal without explosion.
    col = A.sum(axis=0, keepdims=True)
    scale = np.minimum(1.0, 0.93 / np.clip(col, 1e-6, None))
    return A * scale


def _simulate_narrative(reach, bridge, gdelt_volume, total_seconds, bin_seconds, rng):
    """Simulate one narrative's event stream. Returns list of (t, platform)."""
    A = _branching_matrix(bridge)
    events = []
    queue = []

    # Immigrants (exogenous arrivals). News immigrants are modulated by the
    # exogenous GDELT volume signal so that mu_news(t) tracks GDELT.
    base_immigrants = np.array([1.4, 0.7, 0.5]) * reach
    num_bins = len(gdelt_volume)
    gdelt_p = gdelt_volume / max(gdelt_volume.sum(), 1e-6)

    for p in range(NUM_PLATFORMS):
        k = rng.poisson(base_immigrants[p])
        for _ in range(k):
            if p == 2:
                # sample time bin proportional to GDELT volume, then jitter
                b = rng.choice(num_bins, p=gdelt_p)
                t = min(total_seconds - 1, b * bin_seconds + rng.uniform(0, bin_seconds))
            else:
                t = rng.uniform(0, total_seconds)
            queue.append((t, p))

    # Offspring branching (BFS over the excitation tree)
    while queue:
        t, j = queue.pop()
        events.append((t, j))
        for i in range(NUM_PLATFORMS):
            n_children = rng.poisson(A[i, j])
            for _ in range(n_children):
                tc = t + rng.exponential(OFFSPRING_TAU[i])
                if tc < total_seconds:
                    queue.append((tc, i))
    return events


def generate_synthetic_multiplex_data(
    num_events=10000,   # target scale; controls narrative count
    num_users=1000,
    num_narratives=120,
    msg_dim=BASE_MSG_DIM,
    seed=42,
):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    print(f"Generating Hawkes-branching multiplex data (seed={seed})...")

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    total_seconds = int((end_date - start_date).total_seconds())
    bin_seconds = 15 * 60
    num_bins = int(total_seconds / bin_seconds) + 1

    # Exogenous GDELT volume: AR(1) with sparse shocks.
    gdelt_volume = np.zeros(num_bins)
    gdelt_volume[0] = 10.0
    for b in range(1, num_bins):
        gdelt_volume[b] = max(0.0, 0.85 * gdelt_volume[b - 1] + rng.normal(0, 2))
    spike_bins = rng.choice(num_bins, size=6, replace=False)
    gdelt_volume[spike_bins] += rng.uniform(30, 80, size=6)

    # Communities for homophily; Pareto user influence (super-spreaders).
    num_communities = 5
    user_communities = rng.integers(0, num_communities, size=num_users)
    user_influence = rng.pareto(a=2.0, size=num_users) + 1.0
    narrative_communities = rng.integers(0, num_communities, size=num_narratives)

    # Per-narrative latents: reach (Pareto) and bridge (cross-platform tendency).
    narrative_reach = rng.pareto(a=1.5, size=num_narratives) + 1.0
    narrative_bridge = rng.beta(1.5, 3.0, size=num_narratives)  # skewed toward local

    # Narrative message embeddings encode (reach, bridge) in the first two dims
    # (so the latent is recoverable by the model) plus random structure.
    base_embeddings = rng.standard_normal((num_narratives, msg_dim))
    base_embeddings[:, 0] = np.log1p(narrative_reach)
    base_embeddings[:, 1] = 4.0 * (narrative_bridge - narrative_bridge.mean())

    rows = []
    for n in range(num_narratives):
        ev = _simulate_narrative(
            narrative_reach[n], narrative_bridge[n],
            gdelt_volume, total_seconds, bin_seconds, rng,
        )
        c_n = narrative_communities[n]
        # candidate source users: same community, influence-weighted
        u_cand = np.where(user_communities == c_n)[0]
        if u_cand.size == 0:
            u_cand = np.arange(num_users)
        u_w = user_influence[u_cand]
        u_w = u_w / u_w.sum()
        dst_id = n + num_users
        for (t, p) in ev:
            # 10% cross-community leakage in the source
            if rng.random() < 0.90:
                src = rng.choice(u_cand, p=u_w)
            else:
                src = rng.integers(0, num_users)
            b = min(num_bins - 1, int(t // bin_seconds))
            rows.append({
                'src': int(src),
                'dst': int(dst_id),
                't': float(t),
                'platform': int(p),
                'gdelt_volume': float(gdelt_volume[b]),
                'msg': base_embeddings[n] + rng.normal(0, 0.1, size=msg_dim),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values('t').reset_index(drop=True)
    df['msg'] = list(np.stack(df['msg'].values))
    df['log_engagement'] = compute_future_engagement(df, prediction_window_seconds=86400)

    # Diagnostic: how much of the target's variance is *within-narrative*
    # (temporal / excitation-driven) vs *between-narrative* (static reach)?
    # A high within-narrative fraction means the target rewards temporal models.
    y = df['log_engagement'].to_numpy()
    grand = y.mean()
    within_ss, between_ss = 0.0, 0.0
    for nid, g in df.groupby('dst'):
        yi = g['log_engagement'].to_numpy()
        within_ss += ((yi - yi.mean()) ** 2).sum()
        between_ss += len(yi) * (yi.mean() - grand) ** 2
    within_frac = within_ss / max(within_ss + between_ss, 1e-9)

    print(f"  Total events: {len(df)}")
    print(f"  Narratives: {num_narratives} | Users: {num_users}")
    print(f"  Effective msg dim (after dataset.py concat): {EFFECTIVE_MSG_DIM}")
    print(f"  Platform distribution:\n{df['platform'].value_counts().sort_index()}")
    print(f"  Mean bridge (viral) narratives > 0.4: {(narrative_bridge > 0.4).mean():.2f}")
    print(f"  Within-narrative variance fraction (temporal signal): {within_frac:.3f}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=int, default=10000)
    parser.add_argument('--narratives', type=int, default=120)
    parser.add_argument('--out', type=str, default='synthetic_events.pkl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_multiplex_data(
        num_events=args.events, num_narratives=args.narratives, seed=args.seed
    )
    df.to_pickle(args.out)
    print(f"Saved to {args.out}")
