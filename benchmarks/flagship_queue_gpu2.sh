#!/bin/bash
# Flagship RGCN training queue — GPU 2: the 14 single-factor ablations
# (ph split, seed 42; nostat runs on the GPU-1 queue).
# Launch: CUDA_VISIBLE_DEVICES=2 nohup bash benchmarks/flagship_queue_gpu2.sh \
#           > data/retrain/flagship/queue_gpu2.log 2>&1 &
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

for key in l05c10 l10c10 reg_only cls_only fpw1 fpw4 h32 h128 \
           lr3e4 lr3e3 do00 do03 wd0 wd1e3; do
  run_cfg rgcn/flagship/ablations/config_$key.yml
done
echo "QUEUE_GPU2_DONE $(date '+%F %T')"
