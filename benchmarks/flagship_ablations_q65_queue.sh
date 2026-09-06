#!/bin/bash
# q65 ablation sweep: train the 15 one-factor ablation variants on the q65
# split (arrays already built), then assemble the Table-8-style sweep
# (results/flagship/rgcn_ablation_sweep_q65.md).
#
# Launch (two train shards + a finisher that waits for both):
#   nohup bash benchmarks/flagship_ablations_q65_queue.sh A > results/flagship/ablq65_gpuA.log 2>&1 &
#   nohup bash benchmarks/flagship_ablations_q65_queue.sh B > results/flagship/ablq65_gpuB.log 2>&1 &
#   nohup bash benchmarks/flagship_ablations_q65_queue.sh EVAL > results/flagship/ablq65_eval.log 2>&1 &
# GPU ids are set below per shard.
cd /bluesclues-data/home/michaelmurphy/saas-river-forecasting
export PYTHONUNBUFFERED=1
MARK=data/retrain/flagship

run_train() {
  echo "=== train ablq65 $1  $(date '+%T') ==="
  RGCN_CONFIG=rgcn/flagship/ablations_q65/config_$1.yml \
    uv run python -m rgcn.pipeline.train || echo "TRAIN FAILED: $1"
}

case "$1" in
  A)
    export CUDA_VISIBLE_DEVICES=0
    for c in l05c10 l10c10 reg_only cls_only fpw1 fpw4 h32 h128; do
      run_train $c
    done
    touch $MARK/.ablq65_A_done
    echo "SHARD_A_DONE $(date '+%F %T')"
    ;;
  B)
    export CUDA_VISIBLE_DEVICES=1
    for c in lr3e4 lr3e3 do00 do03 wd0 wd1e3 nostat; do
      run_train $c
    done
    touch $MARK/.ablq65_B_done
    echo "SHARD_B_DONE $(date '+%F %T')"
    ;;
  EVAL)
    until [ -f $MARK/.ablq65_A_done ] && [ -f $MARK/.ablq65_B_done ]; do
      sleep 60
    done
    echo "=== both shards done; running sweep eval  $(date '+%T') ==="
    CUDA_VISIBLE_DEVICES=0 ABL_FAMILY=q65 \
      uv run python benchmarks/flagship_ablation_eval.py || echo "EVAL FAILED"
    echo "ABLQ65_QUEUE_DONE $(date '+%F %T')"
    ;;
  *) echo "usage: $0 A|B|EVAL"; exit 1 ;;
esac
