#!/usr/bin/env bash
# Revision compute queue: Step 2 (SIR benchmark), Step 3 (extra seeds),
# Step 5 (Kendall loss-weight sweep). Sequential to avoid CPU contention.
set -u
COMMON="SIMCITY_EPOCHS=15 SIMCITY_EXC_DIM=0"

echo "=== STEP 2: non-Hawkes SIR benchmark, 3 seeds ==="
for s in 1 2 3; do
  env $COMMON SIMCITY_DATA=data/sir_events.pkl SIMCITY_TGN_W=0.1 \
    SIMCITY_HAWKES_W=10 SIMCITY_VIRALITY_W=0.1 \
    SIMCITY_SEED=$s SIMCITY_TAG=sir_s$s python -u train.py > results/log_sir_s$s.log 2>&1
  echo "  sir seed $s exit=$?"
done

echo "=== STEP 3: extra seeds on the real corpus (4-8) ==="
for s in 4 5 6 7 8; do
  env $COMMON SIMCITY_DATA=data/real_events.pkl SIMCITY_TGN_W=0.1 \
    SIMCITY_HAWKES_W=10 SIMCITY_VIRALITY_W=0.1 \
    SIMCITY_SEED=$s SIMCITY_TAG=real_s$s python -u train.py > results/log_real_s$s.log 2>&1
  echo "  real seed $s exit=$?"
done

echo "=== STEP 5: Kendall loss-weight sweep (synthetic) ==="
for w in 1 3 30; do
  env $COMMON SIMCITY_DATA=data/synthetic_events.pkl SIMCITY_TGN_W=0.1 \
    SIMCITY_HAWKES_W=$w SIMCITY_VIRALITY_W=0.1 \
    SIMCITY_SEED=1 SIMCITY_TAG=kw$w python -u train.py > results/log_kw$w.log 2>&1
  echo "  hawkes_w=$w exit=$?"
done
echo "=== QUEUE COMPLETE ==="
