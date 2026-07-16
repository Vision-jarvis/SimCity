"""Count test-split transfer cases in a dataset — the statistical power gate
for the real-data transfer evaluation.

A "transfer case" is a narrative whose test-window events start on one platform
and later switch to another (the positive class in narrative_transfer_eval.py).
The real-data transfer test is underpowered below ~150 positives; run this
after accumulation pulls to decide when to retrain and re-evaluate.

Usage:
    python scripts/count_transfer_cases.py [data/real_events.pkl]
"""

import sys

import numpy as np
import pandas as pd


def count(path):
    df = pd.read_pickle(path).sort_values("t").reset_index(drop=True)
    t = df["t"].to_numpy(float)
    t_min, t_max = t.min(), t.max()
    test = t >= t_min + 0.80 * (t_max - t_min)
    d = df[test]

    n_narr, n_transfer = 0, 0
    for _, g in d.groupby("dst"):
        p = g.sort_values("t")["platform"].to_numpy()
        if len(p) == 0:
            continue
        # narratives with at least one pre-switch event (or no switch) count;
        # positives are those that switch platform within the test window
        switch = np.where(p != p[0])[0]
        if switch.size and switch[0] == 0:
            continue
        n_narr += 1
        if switch.size:
            n_transfer += 1
    return n_narr, n_transfer, int(test.sum()), len(df)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_events.pkl"
    n_narr, n_transfer, n_test, n_all = count(path)
    print(f"dataset: {path} ({n_all} events, {n_test} in test split)")
    print(f"test narratives: {n_narr} | transfer cases: {n_transfer}")
    print(f"power gate (~150 transfer cases): {'PASS' if n_transfer >= 150 else 'below threshold'}")
