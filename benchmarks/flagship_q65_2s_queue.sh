#!/bin/bash
# Two-sided-imputation q65 retrain: splits/arrays now (CPU), then 3 seeds of
# train -> export -> eval_report -> stride-1 day-3 export on GPU 1 as soon as
# ablation shard B releases it.
# Launch: nohup bash benchmarks/flagship_q65_2s_queue.sh > results/flagship/q652s_queue.log 2>&1 &
cd /bluesclues-data/home/michaelmurphy/saas-river-forecasting
export PYTHONUNBUFFERED=1

echo "=== make_splits + prepare_data (2s)  $(date '+%T') ==="
RGCN_CONFIG=rgcn/flagship/config_q65_2s.yml uv run python -m rgcn.pipeline.make_splits \
  || { echo "SPLITS FAILED"; exit 1; }
RGCN_CONFIG=rgcn/flagship/config_q65_2s.yml uv run python -m rgcn.pipeline.prepare_data \
  || { echo "PREPARE FAILED"; exit 1; }

echo "=== waiting for ablation shard B to release GPU 1  $(date '+%T') ==="
until [ -f data/retrain/flagship/.ablq65_B_done ]; do sleep 60; done
export CUDA_VISIBLE_DEVICES=1

for s in "" _s43 _s44; do
  cfg=rgcn/flagship/config_q65_2s$s.yml
  echo "=== train 2s$s  $(date '+%T') ==="
  RGCN_CONFIG=$cfg uv run python -m rgcn.pipeline.train || { echo "TRAIN FAILED: $s"; continue; }
  RGCN_CONFIG=$cfg uv run python -m rgcn.pipeline.export_predictions || echo "EXPORT FAILED: $s"
  RGCN_CONFIG=$cfg uv run python -m rgcn.pipeline.eval_report || echo "EVAL FAILED: $s"
  echo "=== stride-1 export 2s$s  $(date '+%T') ==="
  RGCN_CONFIG=$cfg uv run python -m rgcn.pipeline.export_predictions \
    --eval-stride 1 --day3-range "2020-09-11:2020-12-31" || echo "S1 EXPORT FAILED: $s"
done
echo "Q652S_QUEUE_DONE $(date '+%F %T')"
