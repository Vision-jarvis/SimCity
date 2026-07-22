"""Second synthetic benchmark: a NON-Hawkes generative mechanism.

Revision Step 2. The primary benchmark generates events from a per-narrative
cross-platform Hawkes branching process --- i.e. from the same family the model
assumes. A reviewer will (correctly) ask whether the transfer result is an
artifact of that match. This generator produces the transfer phenomenon by a
structurally different mechanism:

  * Agents live on a **multiplex social graph**. Most belong to one platform;
    a minority are *bridge* users present on both.
  * Each narrative spreads by discrete-time **SIR contagion over the graph**:
    susceptible neighbours of infected users become infected with probability
    beta, infected users recover with probability gamma. There is no excitation
    kernel, no self-exciting intensity, no branching ratio.
  * **Cross-platform transfer is topological, not temporal**: it happens only
    when the contagion reaches a bridge user who then cross-posts (with
    per-narrative propensity b_n). Transfer is therefore driven by *who* the
    narrative reaches, not by *when* events cluster.

If the bridge score still ranks transfer-prone narratives here, the capability
is not an artifact of assuming a Hawkes data-generating process.

Output schema is identical to the Hawkes generator, so the whole pipeline and
evaluation suite run unchanged.

Usage:
    python -m data.sir_multiplex_generator --narratives 320 --out data/sir_events.pkl
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from data.synthetic_generator import compute_future_engagement, BASE_MSG_DIM

NUM_PLATFORMS = 3  # 0 = platform A, 1 = platform B, 2 = news/aggregator


def build_multiplex_graph(num_users, rng, mean_degree=8, bridge_frac=0.32):
    """Two platform communities plus a bridge population present on both.

    Returns adjacency lists and each user's platform membership mask.
    """
    plat_of = np.where(rng.random(num_users) < 0.5, 0, 1)
    is_bridge = rng.random(num_users) < bridge_frac
    # Bridge users additionally participate on the *other* platform.
    alt_plat = np.where(plat_of == 0, 1, 0)

    # Preferential-attachment-ish within-platform wiring (heavy-tailed degree).
    adj = [[] for _ in range(num_users)]
    for p in (0, 1):
        members = np.where((plat_of == p) | (is_bridge & (alt_plat == p)))[0]
        if len(members) < 3:
            continue
        deg = np.ones(len(members))
        for i, u in enumerate(members):
            k = max(1, int(rng.poisson(mean_degree / 2)))
            probs = deg / deg.sum()
            targets = rng.choice(len(members), size=min(k, len(members) - 1),
                                 replace=False, p=probs)
            for tj in targets:
                v = members[tj]
                if v == u:
                    continue
                adj[u].append(v)
                adj[v].append(u)
                deg[i] += 1
                deg[tj] += 1
    adj = [np.array(sorted(set(a)), dtype=np.int64) for a in adj]
    return adj, plat_of, is_bridge, alt_plat


def simulate_sir(seed_user, seed_plat, adj, plat_of, is_bridge, alt_plat,
                 beta, gamma, bridge_prop, rng, max_steps=11):
    """Discrete-time SIR on the graph. Returns [(step, user, platform)]."""
    infected = {seed_user: seed_plat}
    recovered = set()
    events = [(0, seed_user, seed_plat)]
    for step in range(1, max_steps):
        if not infected:
            break
        new_infected = {}
        for u, up in list(infected.items()):
            for v in adj[u]:
                if v in infected or v in recovered or v in new_infected:
                    continue
                # contact only counts if v participates on the poster's platform
                v_on = (plat_of[v] == up) or (is_bridge[v] and alt_plat[v] == up)
                if v_on and rng.random() < beta:
                    new_infected[v] = up
        for v, vp in new_infected.items():
            events.append((step, v, vp))
            # topological cross-posting: only bridge users can move a narrative
            if is_bridge[v] and rng.random() < bridge_prop:
                events.append((step, v, alt_plat[v]))
        # recovery
        for u in list(infected):
            if rng.random() < gamma:
                del infected[u]
                recovered.add(u)
        infected.update(new_infected)
    return events


def generate(num_users=1500, num_narratives=320, msg_dim=BASE_MSG_DIM, seed=42,
             step_seconds=3 * 3600):
    rng = np.random.default_rng(seed)
    print(f"Building multiplex graph ({num_users} users)...")
    adj, plat_of, is_bridge, alt_plat = build_multiplex_graph(num_users, rng)
    print(f"  bridge users: {is_bridge.sum()} ({is_bridge.mean():.0%})")

    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 31)
    total_seconds = int((end - start).total_seconds())

    # Per-narrative latents: infectivity and (crucially) bridging propensity.
    # beta/gamma are kept sub-explosive so cascades stay comparable in size to
    # the Hawkes benchmark rather than saturating the graph.
    narrative_beta = rng.uniform(0.035, 0.095, num_narratives)
    narrative_bridge = rng.beta(2.0, 2.0, num_narratives)   # transfer propensity
    gamma_rec = 0.58

    base_emb = rng.standard_normal((num_narratives, msg_dim))
    base_emb[:, 0] = 4.0 * (narrative_beta - narrative_beta.mean())
    base_emb[:, 1] = 4.0 * (narrative_bridge - narrative_bridge.mean())

    rows = []
    for n in range(num_narratives):
        seed_user = int(rng.integers(0, num_users))
        seed_plat = int(plat_of[seed_user])
        t0 = rng.uniform(0, total_seconds * 0.97)
        ev = simulate_sir(seed_user, seed_plat, adj, plat_of, is_bridge,
                          alt_plat, narrative_beta[n], gamma_rec,
                          narrative_bridge[n], rng)
        for (step, user, p) in ev:
            t = t0 + step * step_seconds + rng.uniform(0, step_seconds * 0.4)
            if t >= total_seconds:
                continue
            rows.append({"src": int(user), "dst": int(n + num_users),
                         "t": float(t), "platform": int(p),
                         "msg": base_emb[n] + rng.normal(0, 0.1, msg_dim)})

    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)

    # Exogenous volume proxy: global event rate per 15-min bin.
    bin_s = 15 * 60
    b = ((df["t"] - df["t"].min()) // bin_s).astype(int)
    df["gdelt_volume"] = b.map(b.value_counts()).astype(float).values

    df["msg"] = list(np.stack(df["msg"].values))
    df["log_engagement"] = compute_future_engagement(df, prediction_window_seconds=86400)

    span = df.groupby("dst")["platform"].nunique()
    y = df["log_engagement"].to_numpy()
    grand = y.mean()
    w = bt = 0.0
    for _, g in df.groupby("dst"):
        yi = g["log_engagement"].to_numpy()
        w += ((yi - yi.mean()) ** 2).sum()
        bt += len(yi) * (yi.mean() - grand) ** 2

    print(f"  events: {len(df)} | narratives: {num_narratives}")
    print(f"  platform distribution:\n{df['platform'].value_counts().sort_index()}")
    print(f"  cross-platform narratives: {(span > 1).sum()} / {num_narratives}")
    print(f"  within-narrative variance fraction: {w / max(w + bt, 1e-9):.3f}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--narratives", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/sir_events.pkl")
    args = ap.parse_args()
    df = generate(num_users=args.users, num_narratives=args.narratives, seed=args.seed)
    df.to_pickle(args.out)
    print(f"Saved to {args.out}")
