# Canonical RGCN training

The model uses 30 days of history, a three-day forecast tail, 36 features including status availability and 17 statics, and the September 10, 2020 temporal cutoff. During the forecast tail, observation and meteorological inputs are held at values available at issue time. Classification uses dry-only discharge augmentation; regression uses observed discharge targets.

From the repository root, after downloading the raw inputs:

```bash
export RGCN_CONFIG=rgcn/configs/seed42.yml
uv run python -m rgcn.pipeline.make_splits
uv run python -m rgcn.pipeline.prepare_data
uv run python -m rgcn.pipeline.build_graph
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.train
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.export_predictions --eval-stride 1 --day3-range "2020-09-11:2020-12-31"
```

Training requires a compatible NVIDIA GPU. Repeat training and export with `seed43.yml` and `seed44.yml`; the prepared arrays and graph are shared. Exports and checkpoints are placed under `data/retrain/paper/`. The checked-in `results/paper/predictions/rgcn_seed*_day*.csv` files are snapshots of the original canonical runs. New runs can vary across hardware; do not overwrite the paper snapshots without reviewing their metrics.

The twelve alternatives under `rgcn/configs/sensitivity/`, together with the default seed-42 model, are the paper's 13 sensitivity rows. Set `RGCN_CONFIG` to each configuration, then train and export on the same daily grid. No split or feature-array rebuild is needed for these settings. `results/paper/sensitivity.csv` records their classification and observed-discharge metrics.

Classification evaluations use the matched sensor dates. Discharge NSE/KGE are computed on observed discharge targets after inverse log(1 + discharge), with 1,038 Day-1 and 1,054 Day-3 observations. Temporal validation was also used for RGCN checkpoint selection and development, so it is not an untouched final test set.
