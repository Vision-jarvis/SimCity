#!/usr/bin/env bash
set -u
C="SIMCITY_EPOCHS=15 SIMCITY_EXC_DIM=0 SIMCITY_TGN_W=0.1 SIMCITY_HAWKES_W=10 SIMCITY_VIRALITY_W=0.1"
for s in 1 2 3; do
  env $C SIMCITY_DATA=data/real_events_emb.pkl SIMCITY_SEED=$s SIMCITY_TAG=emb_s$s \
    python -u train.py > results/log_emb_s$s.log 2>&1
  echo "emb s$s exit=$?"
done
echo "=== CORRECTED CORPUS COMPLETE ==="
