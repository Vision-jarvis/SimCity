#!/usr/bin/env bash
# Retrain on the CORRECTED real corpus (fixed clustering + sentence-embedding
# content features). This is the decisive re-run of the real-data transfer
# result over validated narratives.
set -u
for s in 1 2 3; do
  env SIMCITY_EPOCHS=15 SIMCITY_EXC_DIM=0 SIMCITY_DATA=data/real_events_emb.pkl \
    SIMCITY_TGN_W=0.1 SIMCITY_HAWKES_W=10 SIMCITY_VIRALITY_W=0.1 \
    SIMCITY_SEED=$s SIMCITY_TAG=emb_s$s python -u train.py > results/log_emb_s$s.log 2>&1
  echo "emb seed $s exit=$?"
done
echo "=== CORRECTED CORPUS QUEUE COMPLETE ==="
