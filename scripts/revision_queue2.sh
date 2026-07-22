#!/usr/bin/env bash
# Revised queue. The original queued extra seeds on data/real_events.pkl, whose
# narratives were compromised by the clustering bug -- those runs were dropped.
# Priority is now: finish the SIR benchmark, then the CORRECTED real corpus,
# then the Kendall sweep.
set -u
C="SIMCITY_EPOCHS=15 SIMCITY_EXC_DIM=0 SIMCITY_TGN_W=0.1 SIMCITY_VIRALITY_W=0.1"

echo "=== SIR seed 3 (restart) ==="
env $C SIMCITY_DATA=data/sir_events.pkl SIMCITY_HAWKES_W=10 \
  SIMCITY_SEED=3 SIMCITY_TAG=sir_s3 python -u train.py > results/log_sir_s3.log 2>&1
echo "  sir s3 exit=$?"

echo "=== CORRECTED real corpus, 3 seeds (decisive re-run) ==="
for s in 1 2 3; do
  env $C SIMCITY_DATA=data/real_events_emb.pkl SIMCITY_HAWKES_W=10 \
    SIMCITY_SEED=$s SIMCITY_TAG=emb_s$s python -u train.py > results/log_emb_s$s.log 2>&1
  echo "  emb s$s exit=$?"
done

echo "=== Kendall loss-weight sweep (synthetic) ==="
for w in 1 3 30; do
  env $C SIMCITY_DATA=data/synthetic_events.pkl SIMCITY_HAWKES_W=$w \
    SIMCITY_SEED=1 SIMCITY_TAG=kw$w python -u train.py > results/log_kw$w.log 2>&1
  echo "  kw=$w exit=$?"
done
echo "=== QUEUE 2 COMPLETE ==="
