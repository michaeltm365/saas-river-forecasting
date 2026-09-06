#!/bin/bash
# Flagship analysis queue: stride-1 day-3 exports (12) -> held-out-site
# scoring (3 seeds) -> ablation sweep table -> persistence / matched
# head-to-head / copula.
# Launch: CUDA_VISIBLE_DEVICES=1 nohup bash benchmarks/flagship_analysis_queue.sh \
#           > results/flagship/analysis_queue.log 2>&1 &
cd /bluesclues-data/home/michaelmurphy/saas-river-forecasting
export PYTHONUNBUFFERED=1

export_s1() {  # config, day3-range
  echo "=== stride-1 export: $1 ($2)  $(date '+%T') ==="
  RGCN_CONFIG=$1 uv run python -m rgcn.pipeline.export_predictions \
    --eval-stride 1 --day3-range "$2" || echo "EXPORT FAILED: $1"
}

SEASON="2020-06-16:2020-10-29"
for s in "" _s43 _s44; do
  export_s1 rgcn/flagship/config_ph$s.yml  "$SEASON"
  export_s1 rgcn/flagship/config_sh$s.yml  "$SEASON"
  export_s1 rgcn/flagship/config_q65$s.yml "2020-09-11:2020-12-31"
  export_s1 rgcn/flagship/config_q80$s.yml "2020-09-29:2020-12-31"
done

for s in "" _s43 _s44; do
  tag="flag_sh${s/#_s/_s}"
  echo "=== holdout scoring: $tag  $(date '+%T') ==="
  RGCN_CONFIG=rgcn/flagship/config_sh$s.yml uv run python -m rgcn.pipeline.eval_holdout_sites \
    --predictions-dir data/retrain/flagship/predictions_${tag}_stride1 \
    || echo "HOLDOUT SCORING FAILED: $tag"
done

echo "=== ablation sweep table  $(date '+%T') ==="
uv run python benchmarks/flagship_ablation_eval.py || echo "ABLATION EVAL FAILED"

echo "=== persistence / matched / copula  $(date '+%T') ==="
uv run python benchmarks/flagship_analysis.py || echo "ANALYSIS FAILED"

echo "ANALYSIS_QUEUE_DONE $(date '+%F %T')"
