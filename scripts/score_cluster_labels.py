"""Score human cluster-precision labels and update the paper's number.

After you fill the `same_event_human` column in
results/cluster_labelling_corrected.csv (1 = the HN story and news article are
the same news event; 0 = not), run this to compute precision overall and by
cosine band, and to write results/cluster_precision.json.

    python scripts/score_cluster_labels.py

It also prints agreement with the LLM pre-labels so you can see which pairs to
double-check.
"""

import json

import pandas as pd

CSV = "results/cluster_labelling_corrected.csv"


def main():
    df = pd.read_csv(CSV)
    col = "same_event_human"
    if col not in df.columns or df[col].astype(str).str.strip().eq("").all():
        raise SystemExit(
            f"No human labels found in '{col}'. Fill that column with 1/0 "
            f"in {CSV} first (see the labelling instructions)."
        )
    h = pd.to_numeric(df[col], errors="coerce")
    labelled = h.notna()
    if labelled.sum() < len(df):
        print(f"[warn] only {labelled.sum()}/{len(df)} rows labelled; "
              "scoring the labelled subset.")
    d = df[labelled].copy()
    d["h"] = h[labelled].astype(int)

    prec = float(d["h"].mean())
    n1, n = int(d["h"].sum()), int(len(d))
    print(f"HUMAN cluster precision (same-event): {n1}/{n} = {prec:.1%}")

    bands = [(0.0, 0.55), (0.55, 0.65), (0.65, 1.01)]
    band_out = {}
    for lo, hi in bands:
        m = (d["cosine"] >= lo) & (d["cosine"] < hi)
        if m.sum():
            p = float(d[m]["h"].mean())
            band_out[f"[{lo:.2f},{hi:.2f})"] = {"precision": p, "n": int(m.sum())}
            print(f"  cosine [{lo:.2f},{hi:.2f}): {p:.0%} (n={m.sum()})")

    if "same_event" in d.columns:
        agree = float((d["h"] == d["same_event"]).mean())
        print(f"  agreement with LLM pre-labels: {agree:.0%}")
        disagree = d[d["h"] != d["same_event"]]
        if len(disagree):
            print("  pairs where you disagreed with the LLM (rows, 1-indexed):",
                  [int(i) + 2 for i in disagree.index])  # +2 -> spreadsheet row

    json.dump({"n_pairs": n, "n_same_event": n1, "precision": prec,
               "by_cosine": band_out, "adjudicator": "human"},
              open("results/cluster_precision.json", "w"), indent=2)
    print("\nWrote results/cluster_precision.json (adjudicator = human).")
    print("Then in paper/main.tex, update the two precision mentions "
          "(results (4) and limitation (v)) and change "
          "'LLM-assisted' -> 'human-labelled'.")


if __name__ == "__main__":
    main()
