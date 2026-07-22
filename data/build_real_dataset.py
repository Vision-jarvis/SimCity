"""Build a REAL multiplex dataset from live Hacker News (Algolia) data, in the
exact schema the SimCity pipeline consumes (src, dst, t, platform, gdelt_volume,
msg, log_engagement).

Reality check: in this environment only HN Algolia is reachable without API keys
(Reddit needs OAuth -> 403; GDELT -> 429; Bluesky blocked). So this is a
single-source real dataset used as a proof-of-concept that the pipeline and the
cross-platform-transfer evaluation run on genuine data. "Platform" here is the
story's URL domain-community (a real, observable partition that carries genuine
temporal structure); narratives are title token-Jaccard topic clusters, which can
span multiple domain-communities -> real cross-platform events per narrative.
Swap in the Reddit/GDELT ingesters (keys required) for the full 3-source graph.

Everything downstream (train.py, *_eval.py) works unchanged.

Usage:
    python data/build_real_dataset.py --pages 4 --out data/real_events.pkl
"""

import argparse
import hashlib
import re
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from data.synthetic_generator import compute_future_engagement, BASE_MSG_DIM

ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date"
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
NUM_PLATFORMS = 3
HEADERS = {"User-Agent": "Mozilla/5.0 (simcity-research)"}

# Cross-SOURCE platform scheme (real narrative transfer): 0 = Hacker News,
# 1 = GDELT news. Narratives (topic clusters) that appear on both sources give
# genuine cross-platform excitation — exactly what the Hawkes model targets.
PLAT_HN, PLAT_NEWS = 0, 1

# Domain-community -> platform id. Hash fallback keeps platforms balanced.
NEWS = {"nytimes.com", "bbc.com", "bbc.co.uk", "theguardian.com", "wsj.com",
        "reuters.com", "bloomberg.com", "washingtonpost.com", "cnbc.com",
        "theverge.com", "techcrunch.com", "arstechnica.com", "wired.com"}
SOCIAL = {"github.com", "twitter.com", "x.com", "youtube.com", "medium.com",
          "reddit.com", "substack.com", "linkedin.com", "gitlab.com"}

TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
STOP = set("the a an and or for with from this that have has are was you your our "
           "how why what when who new now use using can will get got make made "
           "just like about into over out its it's they them their we us via".split())


def platform_of(url):
    if not url:
        return 2  # Ask/Show HN, self-posts -> "other/community"
    dom = urlparse(url).netloc.lower().replace("www.", "")
    if dom in NEWS:
        return 1
    if dom in SOCIAL:
        return 0
    # stable hash bucket over the remaining long tail
    hb = int(hashlib.md5(dom.encode()).hexdigest(), 16) % 3
    return hb


def tokens(title):
    return {w for w in TOKEN_RE.findall((title or "").lower()) if w not in STOP}


def hash_embedding(title, dim=BASE_MSG_DIM):
    """Deterministic offline bag-of-tokens hashing embedding (no model download)."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in tokens(title):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def fetch_hn(pages, hits=1000):
    """Page backwards through recent HN stories via created_at_i cursor."""
    rows, cursor = [], int(time.time())
    for _ in range(pages):
        url = (f"{ALGOLIA}?tags=story&hitsPerPage={hits}"
               f"&numericFilters=created_at_i<{cursor}")
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        hitlist = r.json().get("hits", [])
        if not hitlist:
            break
        for h in hitlist:
            if h.get("created_at_i") and h.get("author"):
                rows.append(h)
        cursor = min(h["created_at_i"] for h in hitlist) - 1
        time.sleep(0.3)
    return rows


def top_keywords(titles, k=18):
    """Most frequent non-stopword tokens across HN titles -> GDELT queries."""
    from collections import Counter
    c = Counter()
    for t in titles:
        c.update(tokens(t))
    return [w for w, _ in c.most_common(k)]


def _parse_seendate(s):
    """GDELT seendate 'YYYYMMDDTHHMMSSZ' -> epoch seconds."""
    from datetime import datetime, timezone
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def fetch_gdelt(keywords, per_query=250, timespan="1w"):
    """Pull recent news articles per keyword (throttled >=6s for GDELT's limit)."""
    seen, rows = set(), []
    for kw in keywords:
        params = {"query": kw, "mode": "artlist", "format": "json",
                  "maxrecords": per_query, "timespan": timespan}
        for attempt in range(5):
            try:
                r = requests.get(GDELT, params=params, headers=HEADERS, timeout=25)
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    arts = r.json().get("articles", [])
                    for a in arts:
                        u = a.get("url")
                        ts = _parse_seendate(a.get("seendate", ""))
                        if u and ts and u not in seen:
                            seen.add(u)
                            rows.append({"author": urlparse(u).netloc.replace("www.", ""),
                                         "title": a.get("title", "") or "",
                                         "url": u, "t": float(ts), "points": 0})
                    break
                time.sleep(6)
            except Exception:
                time.sleep(6)
        time.sleep(6)  # stay under GDELT's 1-req/5s limit
    print(f"  fetched {len(rows)} GDELT news articles ({len(keywords)} topics)")
    return rows


def encode_titles(titles, model_name="all-MiniLM-L6-v2", batch_size=128):
    """Sentence-embed titles (revision Step 7).

    Replaces the lexical hashing representation with real semantic embeddings,
    used both for narrative clustering and as the per-event content features.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(list(titles), batch_size=batch_size,
                       show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def cluster_narratives_embedding(emb, times=None, threshold=0.55,
                                 window_days=7.0):
    """Time-windowed greedy centroid clustering on sentence-embedding cosine.

    Two properties matter for correctness, both learned the hard way:

    * **No forced assignment.** An earlier version capped the number of
      clusters and assigned every subsequent item to its nearest centroid
      *regardless of similarity*, which silently destroyed cluster integrity
      once the cap was reached (most items ended up in unrelated clusters). An
      item that matches nothing above ``threshold`` now starts its own cluster.
    * **Temporal locality.** A narrative is a time-bounded event, so only
      clusters active within ``window_days`` are candidates. This is both more
      faithful and much faster, since the active centroid set stays small.
    """
    n = len(emb)
    if times is None:
        times = np.arange(n, dtype=float) * 0.0
    order = np.argsort(times)
    win = window_days * 86400.0

    centroids, counts, last_t = [], [], []
    active = []                      # indices of clusters inside the window
    assign = np.zeros(n, dtype=np.int64)

    for i in order:
        v, t = emb[i], times[i]
        active = [j for j in active if t - last_t[j] <= win]
        j_best, s_best = -1, -1.0
        if active:
            C = np.stack([centroids[j] for j in active])
            sims = C @ v
            k = int(np.argmax(sims))
            j_best, s_best = active[k], float(sims[k])
        if j_best >= 0 and s_best >= threshold:
            c = counts[j_best]
            centroids[j_best] = (centroids[j_best] * c + v) / (c + 1)
            centroids[j_best] /= np.linalg.norm(centroids[j_best]) + 1e-12
            counts[j_best] = c + 1
            last_t[j_best] = t
            assign[i] = j_best
        else:
            centroids.append(v.copy())
            counts.append(1)
            last_t.append(t)
            j = len(centroids) - 1
            active.append(j)
            assign[i] = j
    return assign, len(centroids)


def cluster_narratives(titles, threshold=0.34, max_narr=400):
    """Greedy token-Jaccard topic clustering -> narrative id per story."""
    narr_tokens, assign = [], []
    for tt in titles:
        tks = tokens(tt)
        best, best_j = -1, 0.0
        for i, nt in enumerate(narr_tokens):
            inter = len(tks & nt)
            if inter:
                j = inter / len(tks | nt)
                if j > best_j:
                    best_j, best = j, i
        if best_j >= threshold and best >= 0:
            assign.append(best)
            narr_tokens[best] |= tks
        elif len(narr_tokens) < max_narr:
            narr_tokens.append(set(tks))
            assign.append(len(narr_tokens) - 1)
        else:  # fall back to nearest even if below threshold
            assign.append(best if best >= 0 else 0)
    return np.array(assign), len(narr_tokens)


def build(pages, out, with_gdelt=True, gdelt_topics=45, gdelt_timespan="1w",
          cluster_method="jaccard", emb_threshold=0.55):
    print("Fetching real Hacker News stories (Algolia)...")
    hits = fetch_hn(pages)
    print(f"  fetched {len(hits)} stories")
    if len(hits) < 100:
        raise RuntimeError("Too few stories fetched; check network / increase --pages")

    hn = pd.DataFrame({
        "author": [h["author"] for h in hits],
        "title": [h.get("title", "") or "" for h in hits],
        "url": [h.get("url") for h in hits],
        "t": [float(h["created_at_i"]) for h in hits],
        "points": [h.get("points", 0) or 0 for h in hits],
    })
    hn["platform"] = PLAT_HN

    frames = [hn]
    if with_gdelt:
        print("Fetching real GDELT news on HN topics (throttled)...")
        gd_rows = fetch_gdelt(top_keywords(hn["title"].tolist(), k=gdelt_topics),
                              timespan=gdelt_timespan)
        if gd_rows:
            gd = pd.DataFrame(gd_rows)
            gd["platform"] = PLAT_NEWS
            frames.append(gd)

    new = pd.concat(frames, ignore_index=True)

    # --- Accumulating raw store: merge with previous pulls and dedupe, so the
    # dataset grows across runs (more history => more genuine cross-platform
    # transition events, the scarce resource for the transfer evaluation). ---
    raw_store = out.replace(".pkl", "") + "_raw.pkl"
    import os as _os
    if _os.path.exists(raw_store):
        old = pd.read_pickle(raw_store)
        new = pd.concat([old, new], ignore_index=True)
    # Dedupe: URL when present, else (author,title,t) triple.
    new["_key"] = new.apply(
        lambda r: r["url"] if isinstance(r["url"], str) and r["url"]
        else f"{r['author']}|{r['title']}|{r['t']}", axis=1)
    new = new.drop_duplicates("_key").drop(columns="_key")
    new.to_pickle(raw_store)
    print(f"  raw store: {len(new)} unique events ({raw_store})")

    df = new.sort_values("t").reset_index(drop=True)

    # src = author/domain id; dst = narrative (topic cluster spanning BOTH sources)
    authors = {a: i for i, a in enumerate(df["author"].unique())}
    num_users = len(authors)
    if cluster_method == "embedding":
        print("Encoding titles with sentence embeddings (Step 7)...")
        emb = encode_titles(df["title"].tolist())
        narr, n_narr = cluster_narratives_embedding(emb, threshold=emb_threshold)
    else:
        emb = None
        narr, n_narr = cluster_narratives(df["title"].tolist())
    df["src"] = df["author"].map(authors).astype(int)
    df["dst"] = (narr + num_users).astype(int)

    # Exogenous "volume" proxy: rolling story rate per 15-min bin (stands in for
    # the GDELT exogenous signal until the GDELT ingester is wired with a key).
    bin_s = 15 * 60
    b = ((df["t"] - df["t"].min()) // bin_s).astype(int)
    vol = b.map(b.value_counts()).astype(float)
    df["gdelt_volume"] = vol.values

    if emb is not None:
        df["msg"] = list(emb)          # real semantic content features
    else:
        df["msg"] = [hash_embedding(t) for t in df["title"]]
    df["log_engagement"] = compute_future_engagement(df, prediction_window_seconds=86400)

    keep = ["src", "dst", "t", "platform", "gdelt_volume", "msg", "log_engagement"]
    df[keep].to_pickle(out)

    # How many narratives genuinely span BOTH sources (real cross-platform transfer)?
    span = df.groupby("dst")["platform"].nunique()
    cross = int((span > 1).sum())

    print(f"  users/domains: {num_users} | narratives: {n_narr} | events: {len(df)}")
    print(f"  platform distribution (0=HN, 1=news):\n{df['platform'].value_counts().sort_index()}")
    print(f"  cross-platform narratives (on both HN & news): {cross} / {n_narr}")
    y = df["log_engagement"].to_numpy()
    grand = y.mean()
    w = b_ = 0.0
    for _, g in df.groupby("dst"):
        yi = g["log_engagement"].to_numpy()
        w += ((yi - yi.mean()) ** 2).sum()
        b_ += len(yi) * (yi.mean() - grand) ** 2
    print(f"  within-narrative variance fraction: {w/max(w+b_,1e-9):.3f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--out", default="data/real_events.pkl")
    ap.add_argument("--no-gdelt", action="store_true", help="HN only (skip GDELT)")
    ap.add_argument("--gdelt-topics", type=int, default=45)
    ap.add_argument("--gdelt-timespan", default="1w")
    ap.add_argument("--cluster-method", choices=["jaccard", "embedding"],
                    default="jaccard")
    ap.add_argument("--emb-threshold", type=float, default=0.55)
    args = ap.parse_args()
    build(args.pages, args.out, with_gdelt=not args.no_gdelt,
          gdelt_topics=args.gdelt_topics, gdelt_timespan=args.gdelt_timespan,
          cluster_method=args.cluster_method, emb_threshold=args.emb_threshold)
