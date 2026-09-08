#!/bin/bash
# TWO-SIDED q65 ablation sweep: 15 one-factor variants on the flagq652s
# arrays, then the Day-3 sweep table (rgcn_ablation_sweep_q65_2s.md).
# Launch:
#   nohup bash benchmarks/flagship_ablations_q65_2s_queue.sh A > results/flagship/ablq652s_gpuA.log 2>&1 &
#   nohup bash benchmarks/flagship_ablations_q65_2s_queue.sh B > results/flagship/ablq652s_gpuB.log 2>&1 &
#   nohup bash benchmarks/flagship_ablations_q65_2s_queue.sh EVAL > results/flagship/ablq652s_eval.log 2>&1 &
cd /bluesclues-data/home/michaelmurphy/saas-river-forecasting
export PYTHONUNBUFFERED=1
MARK=data/retrain/flagship

run_train() {
  echo "=== train ablq652s $1  $(date '+%T') ==="
  RGCN_CONFIG=rgcn/flagship/ablations_q65_2s/config_$1.yml \
    uv run python -m rgcn.pipeline.train || echo "TRAIN FAILED: $1"
}

case "$1" in
  A)
    export CUDA_VISIBLE_DEVICES=0
    for c in l05c10 l10c10 reg_only cls_only fpw1 fpw4 h32 h128; do run_train $c; done
    touch $MARK/.ablq652s_A_done; echo "SHARD_A_DONE $(date '+%F %T')" ;;
  B)
    export CUDA_VISIBLE_DEVICES=1
    for c in lr3e4 lr3e3 do00 do03 wd0 wd1e3 nostat; do run_train $c; done
    touch $MARK/.ablq652s_B_done; echo "SHARD_B_DONE $(date '+%F %T')" ;;
  EVAL)
    until [ -f $MARK/.ablq652s_A_done ] && [ -f $MARK/.ablq652s_B_done ]; do sleep 60; done
    echo "=== sweep eval  $(date '+%T') ==="
    CUDA_VISIBLE_DEVICES=0 ABL_FAMILY=q65_2s \
      uv run python benchmarks/flagship_ablation_eval.py || echo "EVAL FAILED"
    echo "ABLQ652S_QUEUE_DONE $(date '+%F %T')" ;;
  *) echo "usage: $0 A|B|EVAL"; exit 1 ;;
esac
