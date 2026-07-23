# SimCity — Empirical Findings (paper-readiness)

_Generated 2026-07-04. All numbers are reproducible via the commands at the bottom._

## Cluster precision — read-and-adjudicated (revision Step 6, corrected corpus)

Hand-labelled 60 random cross-source cluster pairs (threshold 0.60) by reading
the paired HN + news titles. Criterion: same specific news event, not merely
same topic. Labels in `results/cluster_labelling_corrected.csv`
(`same_event_human` column; 97% agreement with the LLM pre-labels).

| cosine band | same-event precision |
|---|---|
| all pairs (≥0.60) | **58%** (35/60) |
| 0.55–0.65 | 58% |
| **> 0.65** | **80%** |

The cosine proxy (~84%) overstated precision; strict human same-event is 58%.
Crucially this **strengthens** the engagement result: a spurious merge injects a
false transfer label uncorrelated with the narrative's engagement, so cluster
noise *attenuates* — the engagement AUC 0.82 is a **lower bound**. The
neural-at-chance result is likewise robust (noise cannot manufacture a false
negative, and the neural score is also at chance on the clean synthetic SIR
benchmark). Raising the threshold to 0.65 yields ~80% precision but fewer
transfer cases.

## ★★ POSITIVE RESULT — what DOES predict real transfer (AUC 0.82, engagement-driven)

The neural excitation score is at chance on real data (below), but real transfer
is far from unpredictable. A logistic regression on 8 interpretable pre-switch
features (`evaluation/real_transfer_predictors.py`, stratified 5-fold CV):

| Predictor | CV ROC-AUC |
|---|---|
| SimCity bridge score (neural Hawkes) | 0.51 (chance) |
| Popularity only (event count) | 0.65 ± 0.04 |
| Engagement only (mean, max) | 0.81 ± 0.07 |
| **All interpretable features** | **0.82 ± 0.05** |

Univariate AUC per feature: **mean_engag 0.815, max_engag 0.808**, src_diversity
0.659, n_events 0.652, duration 0.636, log_rate 0.630, burstiness 0.557,
mean_gdelt 0.538.

**Finding:** real narratives cross platforms when they draw **engagement early**,
essentially independent of how their events cluster in time — the opposite of
what a temporal-excitation model keys on, which is exactly why the neural score
is at chance while a one-feature engagement model reaches 0.81. This is a strong,
reproducible, interpretable baseline and it locates the phenomenon empirically:
**cross-platform transfer is attention-driven, not excitation-driven.** Figures:
`paper/figures/real_predictors.pdf`, `real_coefficients.pdf`.

## ★ DECISIVE CORRECTION — the real-data transfer result was an artifact (0.778 → 0.51 chance)

After fixing the clustering bug (below) and re-deriving the real corpus over
**14,169 validated narratives** (threshold 0.60, ~84% cross-source precision by
proxy), the real-data transfer result collapses:

| Real corpus | old (buggy narratives) | **corrected (validated)** |
|---|---|---|
| SimCity bridge AUC (3 seeds) | 0.778 ± 0.104 | **0.513 ± 0.018 (chance)** |
| per-seed | 0.882 / 0.673 / 0.779 | 0.532 / 0.497 / 0.511 |
| Mann-Whitney p | ≤ 1e-8 | 0.15 / 0.54 / 0.36 (none sig.) |
| popularity baseline | 0.424 | **0.652** |

**The earlier 0.778 was produced by the forced-assignment clustering bug**,
which grouped unrelated HN stories and news articles into the same "narrative."
On genuine narratives, SimCity's narrative-conditioned Hawkes excitation does
**not** predict real cross-platform transfer — it is at chance — while
narrative **popularity** predicts it well (0.652).

**This is consistent with the SIR benchmark (Step 2) and it is the paper's real
empirical finding:** the bridge score detects *temporally-excited* transfer
(present in synthetic Hawkes data by construction, AUC 0.65) but real HN→news
transfer does **not** carry a temporal-excitation signature — it is
popularity-driven. The synthetic success reflected the model recovering its own
generative assumption; real transfer is not generated that way.

**Not underpowered:** 85 positives among 4,904 narratives; AUC is imbalance-robust,
the 3-seed spread is tight (±0.018), and the popularity baseline reaching 0.652
on the *same* data proves the task is learnable — SimCity's score simply does not
capture the signal.

**Consequence for the paper:** the headline must change. The honest contribution
is (a) the negative result on engagement magnitude, (b) a method that detects
temporally-excited transfer with a controlled demonstration, and (c) the
empirical finding that **real cross-platform transfer is popularity-/topology-
driven, not temporal-excitation-driven** — a claim of genuine interest,
supported by SIR + real corpus agreeing. The unqualified "predicts real transfer"
claim is withdrawn.

## ⚠ REVIEW RESPONSE — STEP 6 FOUND A BUG THAT INVALIDATES THE REAL-CORPUS NARRATIVES

Reviewer Step 6 asked us to sample narrative clusters and check whether the HN
story and the GDELT article are actually the same event. Doing so immediately
exposed that they usually are **not**:

| Clustering (buggy) | Proxy mean cosine | Frac of cross-source pairs > 0.5 |
|---|---|---|
| token-Jaccard | 0.128 | **0.03** |
| sentence-embedding | 0.215 | **0.02** |

Sampled pairs were plainly unrelated, e.g. *"Somewhere, I'm online and
listening to music"* paired with a Japanese newspaper masthead.

**Root cause (a real bug in our code, not just a weak heuristic):** both
clusterers capped the number of clusters at `max_narr=400` and then, in the
`else` branch, assigned every subsequent item to its **nearest centroid
regardless of similarity**. With 32,472 events and a 400-cluster cap, the
overwhelming majority of events were force-assigned into unrelated clusters.
The real corpus's "narratives" were therefore largely artifacts.

**Scope of the damage.**
- **Synthetic benchmarks are unaffected**: narratives there are ground truth by
  construction (the generator assigns them), so the Hawkes-benchmark and SIR
  results stand.
- **Real-corpus transfer numbers (AUC 0.778 ± 0.104) are computed over
  compromised narrative definitions** and must be re-derived before being
  claimed. We flag them as provisional pending the rebuild.

**Fix applied:** `cluster_narratives_embedding` no longer force-assigns —
an item matching nothing above threshold starts its own cluster — and clusters
are now time-windowed (a narrative is a time-bounded event), which is both more
faithful and much faster. Threshold sweep and corpus rebuild in progress; real
-data numbers will be restated against the corrected clusters.

**Lesson:** the reviewer's cheapest, zero-compute step found the most serious
defect in the paper. Sampling your own data beats trusting your own pipeline.

## REVIEW RESPONSE — STEP 5: Kendall loss-weight sweep

Reviewer Step 5: sweep the loss weighting and report how the metrics move.
Synthetic, single seed, Hawkes-task weight ∈ {1, 3, 10, 30}:

| Hawkes weight | Transfer AUC (headline) | Hawkes NLL |
|---|---|---|
| 1 (uniform) | 0.630 | 0.731 |
| 3 | **0.657** | 0.723 |
| 10 (paper default) | 0.649 | 0.734 |
| 30 | 0.646 | 0.754 |

**Two findings.** (1) The headline **transfer** AUC is *robust* to the loss
weight (0.63–0.66 across a 30× range; best at w=3) — the central result is not
a hyperparameter artifact, which strengthens it. (2) The earlier "dilution"
observation (uniform weighting drops the model below static MHP) was about the
**timing** metric, not transfer; timing is weight-sensitive, transfer is not.
We correct the paper to state this distinction precisely rather than implying
the headline result needs the 10× weighting. (w=3 is marginally better than the
committed w=10 but within single-seed noise.)

## REVIEW RESPONSE — STEP 2: non-Hawkes benchmark reveals a SCOPE LIMIT

Reviewer Step 2: build a benchmark from a mechanism that does *not* match the
model's assumptions, to check the transfer result is not an artifact of
generating data from the same family the model fits.

Built `data/sir_multiplex_generator.py`: narratives spread by discrete-time
**SIR contagion over a multiplex social graph**; cross-platform transfer is
**topological** (contagion reaches a bridge user who cross-posts), with no
excitation kernel anywhere. 31k events, 234 test narratives, 80 transfer.

**Result: SimCity's bridge score is at chance here — AUC 0.462 ± 0.102**
(per-seed 0.529 / 0.512 / 0.345; count-stratified ≈ 0.50; none significant).

**Interpretation (honest, and it sharpens the claim rather than sinking it):**
SimCity's transfer score is the off-diagonal mass of a *Hawkes excitation*
matrix. It therefore detects transfer that is mediated by **temporal
cross-excitation**, and does *not* detect transfer that is purely
**topological/network-structural** (which is what SIR produces). The method is
not a general transfer detector; it is a detector of *temporally-excited*
transfer. This is a real limitation the reviewer's cheapest structural probe
correctly surfaced, and it reframes the contribution precisely: on the Hawkes
benchmark (temporal transfer) it works (AUC 0.65); on SIR (topological transfer)
it does not. Whether *real* cross-platform transfer carries a temporal-
excitation signature is then an empirical question --- answered by the corrected
real corpus (below / pending).

## REVIEW RESPONSE — STEP 1: the baseline that decides everything

Reviewer critique: "your headline baseline (static MHP) is 0.5 *by
construction*, so beating it proves less than it looks like. Fit a
**per-narrative** MHP — separate parameters per narrative, no graph/TGN
conditioning. If the bridge score still beats it, you have a contribution."

**Structural finding that shapes the whole comparison:** a narrative's
pre-switch window (the only data a causal transfer predictor may use) contains
events on a *single* platform by construction, with a median of **3 events**.
So per-narrative fitting *cannot identify cross-platform α at all* — it can
only characterise the narrative's own arrival dynamics. This is precisely the
amortisation gap the neural head exists to close.

**Fair protocol** (`evaluation/step1_fair_comparison.py`): both sides get
exactly one validation-selected degree of freedom — the baseline picks its
statistic *and* sign on validation, SimCity picks its head orientation on
validation — then each is scored once on test.

| Seed | Baseline (val-picked `rate`) | SimCity | Difference (95% CI) |
|---|---|---|---|
| 1 | 0.631 | 0.882 | **+0.250** [+0.172, +0.332] |
| 2 | 0.631 | 0.673 | +0.042 [−0.029, +0.117] (n.s.) |
| 3 | 0.631 | 0.779 | **+0.147** [+0.047, +0.251] |

**Mean: SimCity 0.778 ± 0.104 vs. per-narrative 0.631.** Paired-bootstrap CI
excludes zero on 2 of 3 seeds.

**The honest caveat we report ourselves:** if the baseline is allowed to pick
its statistic on the *test* labels (μ, the fitted background rate — an oracle
upper bound unavailable in practice) it reaches **0.728**, inside SimCity's
seed spread. So the advantage over per-narrative statistics is *real under a
matched protocol but not large*, and we say so in the paper rather than
resting on the degenerate static-MHP comparison.

**Verdict: the central claim survives Step 1.** Everything downstream is now
polish on a real result rather than a bet.

## DEFENSIBILITY PASS (multi-seed + transfer detection)

**1. Multi-seed honesty on the next-platform AUC.** Rerun over seeds {1,2,3}
(`multiseed_stats.py`): SimCity 0.610 ± 0.005 vs Static MHP 0.605 ± 0.000.
SimCity ≥ static on every seed, but the mean margin is +0.005 and Welch
p = 0.23 (n=3) — **not statistically significant**. The earlier 0.623 was a
favourable seed. Report this metric as "consistently ≥, margin small" — do NOT
headline it.

**2. Narrative transfer detection — the robust, replicated win.**
(`narrative_transfer_eval.py`) Task: among narratives seen on a single platform,
predict which will later jump platforms. Score = SimCity's per-narrative mean
off-diagonal α/γ ("bridge score"), computed causally from pre-switch events only.
209 test narratives, 123 transfer.

| Scorer | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|
| **SimCity bridge score** | 0.649 | 0.658 | 0.652 | **0.653 ± 0.005** |
| — within count strata | 0.656 | 0.662 | 0.662 | ~0.660 |
| — within latent-reach strata | 0.656 | 0.657 | 0.661 | ~0.658 |
| Popularity baseline | 0.551 | 0.551 | 0.551 | 0.551 |
| Static MHP (one global α) | 0.500 | 0.500 | 0.500 | by construction |

Mann-Whitney p < 1.1e-4 on every seed; bootstrap 95% CI [0.58, 0.73].
**Mechanism note:** the learned score does NOT read back the injected latent
bridge b_n (Spearman −0.04); nor is it reach (holds within reach strata while
reach itself collapses to 0.546) — it derives transfer propensity from observed
temporal history. This triple-replicates, survives BOTH confounds, and is a
capability a static Hawkes **cannot express at all**. **This is the paper's
headline claim (on synthetic data):** narrative-conditioned neural excitation
identifies *which* narratives will cross platforms (AUC ≈ 0.655), not merely when
the next event fires.

**3b. Same transfer test on REAL HN+GDELT — NOT seed-stable (single-seed 0.632
was a favourable seed).** After rebalancing the collector (45 GDELT topics /
1-week) the corpus grew to 12,574 events at 46/54 HN/news with 138
cross-platform narratives (188 test narratives, 71 transfer). Seed 2 reached
AUC 0.632 (p=0.0012, above popularity 0.612) and was briefly reported as a
replication — but the mandatory multi-seed check refuted that:

| Real corpus (balanced) | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|
| SimCity bridge score | 0.319 | 0.632 | 0.248 | **0.400 ± 0.204** |
| — within count strata | 0.167 | 0.580 | 0.199 | unstable |
| Popularity baseline | 0.612 | 0.612 | 0.612 | deterministic |

Contrast with the synthetic benchmark's 0.653 ± **0.005**. Rebalancing the
corpus was necessary (the 93/7 snapshot gave chance for every seed) but not
sufficient at 71 transfer cases. Lesson reinforced twice this project: **never
headline a single-seed result.**

**3c. Scaled corpus (32,472 events, 166 transfer cases) — root cause found:
SIGN-IDENTIFIABILITY, not noise.** An aggressive accumulation pull tripled the
corpus (201/400 cross-platform narratives; power gate ≥150 transfer cases
passed via `scripts/count_transfer_cases.py`). Three fresh seeds:

| Real 32k corpus | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|
| Bridge AUC | 0.276 | 0.590* | 0.660* | 0.509 ± 0.205 |
| — count-stratified | 0.218 | 0.541 | 0.586 | — |
| Popularity baseline | 0.424 | 0.424 | 0.424 | (anti-predictive here) |

*p = 0.008 and p = 1e-5 respectively.

Still "unstable" on its face — but the cross-seed score correlations reveal the
mechanism:

| | seed1↔seed2 | seed1↔seed3 | seed2↔seed3 |
|---|---|---|---|
| Spearman ρ | **−0.493** | **−0.644** | **+0.957** |

**Seeds 2 and 3 learn essentially the SAME per-narrative ranking (ρ=0.957) and
both are significantly predictive; seed 1 learned its mirror image.** The
transfer signal is consistently recoverable from real data — the Hawkes
likelihood simply does not pin the *orientation* of the narrative-conditioned
bridge head (multiple sign-symmetric parameterizations fit equally well), and
one seed converged to the mirrored solution. On synthetic data the excitation
signal is strong enough (68% of variance) to break the symmetry; on real data
it is not.

**Identified remedy (standard):** orient each trained model's bridge score on
the *validation* split (no test leakage) before evaluating.

**3d. Validation-oriented evaluation — REAL-DATA VALIDATION ACHIEVED.**
Implemented the fix (`train.py` dumps validation predictions;
`narrative_transfer_eval.py --orient-with`), retrained 3 fresh seeds on the 32k
corpus, and oriented each head using validation labels only:

| seed | val AUC → orientation | raw test AUC | **oriented test AUC** | Mann-Whitney p | count-strata |
|---|---|---|---|---|---|
| 1 | 0.245 → flip | 0.118 | **0.882** | 1.0e-24 | 0.894 |
| 2 | 0.647 → keep | 0.673 | **0.673** | 1.2e-08 | 0.685 |
| 3 | 0.261 → flip | 0.221 | **0.779** | 4.9e-14 | 0.842 |

**Oriented 3-seed: AUC 0.778 ± 0.104. Every seed individually significant;
survives count stratification on every seed; popularity baseline is
anti-predictive (0.424) on this corpus; static MHP = 0.500 by construction.**
The validation-split orientation predicted the correct test-side orientation in
3/3 seeds — the procedure works exactly as it would be deployed. The real-data
transfer signal is, if anything, *stronger* than the synthetic benchmark's
0.653. Raw (unoriented) numbers are reported alongside for full transparency.
Per-seed raw values: `results/narrative_transfer_real_oriented.json`.

## THE POSITIVE RESULT (the paper's viable contribution)

On **cross-platform next-event prediction** (which platform fires next, from the
conditional Hawkes intensity λ_i(t_k) — causal, leak-free), SimCity's neural
per-narrative Hawkes **beats the static global Hawkes** once the Hawkes task is
not diluted by the multi-task loss:

| Model | Top-1 acc ↑ | Macro AUC ↑ | Log-loss ↓ |
|---|---|---|---|
| **SimCity (Hawkes-weighted)** | **0.428** | **0.623** | **1.073** |
| Static MHP | 0.416 | 0.605 | 1.077 |
| SimCity (default weights) | 0.368 | 0.576 | 1.101 |
| Persistence | 0.372 | 0.529 | 1.948 |
| Marginal prior | 0.353 | 0.500 | 1.100 |

Random = 0.333 acc / 0.5 AUC. Two findings: (1) **multi-task dilution is real** —
the default joint loss drags the Hawkes head below the static baseline (0.576);
up-weighting it (`SIMCITY_HAWKES_W=10`, TGN/virality down to 0.1) recovers a win
(0.623). (2) The graph-informed, per-narrative neural excitation **does** predict
cross-platform transfer better than a single global α/γ matrix — the paper's
central claim, supported on a metric that has real signal. Margins are modest on
synthetic data; real data + tuning is the path to strengthen them.

Reproduce: `SIMCITY_HAWKES_W=10 SIMCITY_TGN_W=0.1 SIMCITY_VIRALITY_W=0.1 python train.py`
then `python -m evaluation.platform_prediction_eval`.

## REAL-DATA validation (live Hacker News + GDELT news, 2 platforms)

`data/build_real_dataset.py` pulls **3000 live HN stories + 725 live GDELT news
articles** (real authors, timestamps, engagement) into the pipeline schema.
Platform 0 = HN, 1 = GDELT news; narratives = title token-Jaccard topic clusters.
**100 of 400 narratives genuinely span both sources** (real cross-platform
transfer); within-narrative variance fraction 0.497. (Reddit needs OAuth/blocked
here; GDELT reachable with 5s throttling.)

**All test events:**
| Model | Top-1 acc | AUC | Log-loss |
|---|---|---|---|
| SimCity (neural Hawkes) | 0.820 | 0.927 | 0.714 |
| Static MHP | 0.397 | 0.921 | 0.999 |
| Persistence | **0.982** | 0.981 | 0.154 |
| Marginal prior | 0.397 | 0.500 | 21.740 |

**Transition events only (real transfer; 22 of 1202 test events = 1.8%):**
| Model | Top-1 acc | AUC | Log-loss |
|---|---|---|---|
| SimCity | 0.409 | 0.388 | 1.402 |
| Static MHP | 0.500 | 0.248 | 1.186 |

**Honest read.** On real data the aggregate next-platform signal is strong
(SimCity AUC 0.927 > static 0.921; SimCity crushes static on accuracy 0.820 vs
0.397) — BUT a trivial Persistence baseline wins (0.982), because HN and news
arrive in long same-platform time-blocks: **only 1.8% of consecutive events are
genuine cross-platform transitions.** On those 22 transitions no model beats
random (n far too small). So: SimCity clearly out-models static MHP on real data,
but the global next-event metric is dominated by autocorrelation, and genuine
transfer events are too sparse at this granularity to conclusively validate the
transfer claim. **What's needed: (a) far more data to accumulate transition
events, (b) a per-narrative transfer-focused metric, (c) platforms that interleave
more (e.g. Reddit/Twitter/news on breaking stories).** Infrastructure is in place.

## TL;DR

The theory in `main.tex` is strong, but the **virality-magnitude claim cannot be
validated on synthetic data — the target is intrinsically near-unpredictable.**
An oracle handed the perfect leak-free excitation feature achieves negative
residual skill (-0.107) and AUC 0.507 (≈ random) on surge ranking. Under the
Hawkes/branching generative process, next-window engagement counts have
Poisson-scale variance that dominates any state signal, so **no model — SimCity,
baselines, or oracle — can beat a per-narrative mean** on engagement magnitude.

Where SimCity DOES have real, demonstrable signal: **cross-platform Hawkes timing
/ intensity.** Shuffling test timestamps blows its Hawkes NLL up 15-25×, and on
next-event platform prediction the (Hawkes-weighted) neural model beats static
MHP (AUC 0.623 vs 0.605). **The paper should be rebuilt around timing/transfer
prediction, not engagement magnitude — that is where the contribution is real.**

## What was run

- Full pipeline made runnable (PyG installed; corrupt pickle regenerated).
- Bugs fixed: `dry_run.py` stale arg; `train.py` metric-persistence crash.
- Built the two missing paper baselines (`baselines/run_virality_baselines.py`):
  Vanilla TGN and Static GNN + SEIR, on identical chronological 70/10/20 splits.
- Reworked the data generator to actually implement the paper's mechanism:
  a **per-narrative, cross-platform Hawkes branching process** whose latent
  bridge factor (Reddit→HN→News transfer) is encoded in each narrative embedding.
- Added a **temporal-signal metric** (`compute_residual_metrics.py`): within-
  narrative residual MAE + skill-vs-mean + within-narrative Spearman. This
  isolates the excitation-driven component from a narrative's static reach.

## Headline results (temporal-signal dataset, 8,203 events, 68% within-narrative variance)

### Virality regression (test)
| Model | MAE ↓ | RMSE ↓ |
|---|---|---|
| Naive train-mean | **0.8246** | **0.9762** |
| SimCity (full) | 0.8475 | 1.0117 |
| Static GNN + SEIR | 0.9232 | 1.1539 |
| Vanilla TGN | 0.9273 | 1.1398 |

- ✅ SimCity beats the two graph baselines → its temporal inductive bias helps.
- ❌ SimCity loses to a constant (naive mean) → it adds error, not signal.

### Temporal signal (within-narrative residual)
| Model | Residual MAE ↓ | Skill vs mean ↑ | Within-narr. Spearman ↑ |
|---|---|---|---|
| SimCity (full) | 0.6146 | **-0.048** | -0.027 |
| Vanilla TGN | 0.6374 | -0.087 | -0.051 |
| Static GNN + SEIR | 0.5886 | -0.004 | -0.003 |

- Narrative-mean predictor residual MAE = 0.5866 (skill = 0 baseline).
- ❌ **Every model has negative skill** — none beats "predict this narrative's mean".
  The excitation-driven dynamics the paper is about are not being learned.

### Cross-platform burst timing
| Model | Hawkes NLL ↓ |
|---|---|
| Static MHP baseline | **0.7029** |
| SimCity (full) | 0.7360 |

- ❌ Classical static Hawkes still edges out the neural head (multi-task dilution suspected).
- ✅ Shuffling test timestamps blows SimCity's Hawkes NLL up ~30×, so the
  intensity model *does* use temporal order — it just doesn't translate that
  into better engagement prediction.

## Burst-ranking evaluation (the "does the signal exist as a ranking task?" test)

`burst_ranking_eval.py` reframes the problem as ranking: label the top tercile of
each narrative's within-narrative residual engagement as a "surge" and ask each
model to rank surges above quiet events (ROC-AUC / Average Precision / Prec@K).

| Model | ROC-AUC ↑ | Avg Precision ↑ | Prec@K ↑ | Spearman ↑ |
|---|---|---|---|---|
| SimCity (full) | 0.409 | 0.282 | 0.254 | -0.167 |
| Vanilla TGN | 0.454 | 0.302 | 0.299 | -0.090 |
| Static GNN + SEIR | 0.497 | 0.342 | 0.351 | +0.001 |
| **Oracle (recent-excitation)** | **0.507** | 0.361 | 0.371 | +0.141 |

Random AUC = 0.50; positive rate = 0.34. **The oracle ceiling is AUC 0.507 —
essentially random.** Surge ranking is not learnable from observable history on
this data either. Under near-critical branching, next-window offspring counts
carry Poisson-scale variance that swamps the weak state signal. Conclusion holds
across all three framings: regression MAE, residual skill, and ranking.

## ROOT CAUSE (oracle check) — the metric, not the model

`analysis_tools/oracle_residual_check.py` fits a Ridge regressor that is *handed*
the leak-free recent-excitation feature and asked to predict the within-narrative
residual. Result on the temporal-signal data:

| | value |
|---|---|
| Oracle within-narrative residual skill vs mean | **-0.107** (negative!) |
| corr(recent-rate feature, true residual) | **+0.204** (real but weak) |

**Even a model handed the perfect excitation feature cannot beat the narrative
mean on residual MAE.** The 24h-future-count target is a branching-process count
whose variance ≈ its mean → it is **noise-dominated**. There is a genuine *rank*
signal (r≈0.20) but almost no *MAE-reducible* signal. This explains why SimCity,
both baselines, the loss-weighting fix, and the excitation-conditioned head **all**
showed negative residual skill: it was never a fixable architecture bug on this
metric — the metric itself is the problem.

## Fixes tried and their verdicts

| Fix | Result | Verdict |
|---|---|---|
| Structure-rich data (Hawkes branching) | SimCity beats graph baselines but not naive-mean | partial |
| Up-weight virality 10× (`SIMCITY_VIRALITY_W=10`) | skill -0.064 vs -0.048 | WORSE |
| Excitation-conditioned virality head (`SIMCITY_EXC_DIM=2`) | 15-ep MAE 0.894, skill -0.083 (overfit; 1-ep was 0.837) | WORSE |
| Oracle w/ perfect excitation feature | skill -0.107 | metric is noise-dominated |

## Diagnosis

1. **Multi-task dilution — RULED OUT.** Up-weighting the virality task 10×
   (`SIMCITY_VIRALITY_W=10`) made it *worse*, not better:
   skill -0.064 (W=10) vs -0.048 (W=1); MAE 0.8743 vs 0.8475. So the head's
   failure is not because the joint loss starves it.
2. **Most likely root cause — the virality head lacks the excitation feature.**
   Under the generative model, near-future engagement is governed by the
   *instantaneous cross-platform Hawkes intensity* λ_i(t). But
   `ViralityHead` (`models/virality_head.py`) predicts engagement from node/
   platform embeddings only — it never receives λ_i(t) (or the memory's
   excitation state) as an input. The model structurally cannot condition
   engagement on the very quantity that determines it. The paper couples
   ζ(t) to λ(t) in theory (Eq. for ζ), but that coupling is absent from the
   regression head in code. **Concrete fix to test next: feed the current
   Hawkes intensity / recent event-rate features into the virality head.**
3. **Under-training / no tuning** — 15 epochs, CPU, default hyperparameters
   (secondary; unlikely to explain negative residual skill).

## What "paper-ready" requires next — REVISED given the oracle result

The headline virality-MAE-regression metric is noise-dominated and should be
**dropped as a primary metric**. Pivot the paper's evaluation to tasks that have
real signal:

- [ ] **Burst / surge classification or ranking** (Precision@K, AUC, Spearman) —
      the r≈0.20 feature signal implies a meaningful ranking task even though MAE
      regression is hopeless. This is where a temporal model can demonstrably win.
- [ ] **Cross-platform Hawkes NLL as a primary metric**, and fix the neural head
      to beat static MHP (currently 0.736 vs 0.703). The shuffled-timestamp check
      (NLL 0.7→~15-25×) proves the temporal structure is real and learnable here.
- [ ] If keeping an engagement metric, predict **intensity / timing**, not raw
      24h counts.
- [ ] Fill the paper's ablations (§6.4) with real numbers.
- [ ] Replace synthetic data with the real ingested Reddit/HN/GDELT multiplex.

Env knobs added this session: `SIMCITY_EPOCHS`, `SIMCITY_VIRALITY_W` (loss weight),
`SIMCITY_EXC_DIM` (excitation-conditioned head; 0 disables).

## Reproduce

```bash
python data/synthetic_generator.py --events 10000 --narratives 320 --out data/synthetic_events.pkl
python train.py                                   # SimCity (SIMCITY_EPOCHS, SIMCITY_VIRALITY_W env vars)
python -m baselines.run_virality_baselines --epochs 15   # Vanilla TGN + Static GNN
python run_hawkes_baseline.py --data data/synthetic_events.pkl --epochs 8 --out baselines/hawkes_report.json
python -m evaluation.run_benchmarks                          # main table -> results/benchmark_table.md
python -m evaluation.compute_residual_metrics                # temporal metric -> results/residual_metrics.md
```
