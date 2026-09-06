#!/bin/bash
# Flagship RGCN training queue — GPU 1: the 4-split x 3-seed matrix + nostat.
# Launch: CUDA_VISIBLE_DEVICES=1 nohup bash benchmarks/flagship_queue_gpu1.sh \
#           > data/retrain/flagship/queue_gpu1.log 2>&1 &
cd /bluesclues-data/home/michaelmurphy/saas-river-forecasting
export PYTHONUNBUFFERED=1

run_cfg() {
  local cfg=$1
  echo "=== $cfg  $(date '+%F %T') ==="
  RGCN_CONFIG=$cfg uv run python -m rgcn.pipeline.train \
    || { echo "TRAIN FAILED: $cfg"; return 1; }
  RGCN_CONFIG=$cfg uv run python -m rgcn.pipeline.export_predictions \
    || { echo "EXPORT FAILED: $cfg"; return 1; }
  RGCN_CONFIG=$cfg uv run python -m rgcn.pipeline.eval_report \
    || { echo "EVAL FAILED: $cfg"; return 1; }
}

wait_for() {  # block until prepare_data output exists
  while [ ! -f "$1" ]; do echo "waiting for $1 ..."; sleep 60; done
}

for s in "" _s43 _s44; do run_cfg rgcn/flagship/config_ph$s.yml; done
for s in "" _s43 _s44; do run_cfg rgcn/flagship/config_sh$s.yml; done

wait_for data/retrain/flagship/feature_arrays_flagq65.npz
for s in "" _s43 _s44; do run_cfg rgcn/flagship/config_q65$s.yml; done

wait_for data/retrain/flagship/feature_arrays_flagq80.npz
for s in "" _s43 _s44; do run_cfg rgcn/flagship/config_q80$s.yml; done

run_cfg rgcn/flagship/ablations/config_nostat.yml
echo "QUEUE_GPU1_DONE $(date '+%F %T')"
