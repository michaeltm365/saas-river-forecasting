# Paper results

- `TABLES.md`, `classification_seeds.csv`: availability-model classification by seed, full/matched sensor set, and stream order.
- `baselines/`: causal LR/XGBoost results and temporal HOBO-only LSTM results, with prediction snapshots.
- `sensitivity.csv`: the default and twelve manuscript sensitivity settings, seed 42. `sensitivity_predictions/` contains the sensor and observed-discharge predictions supporting every row.
- `copula/`: raw seed-42 observed-period counts, intervals, and maps.
- `figure3/`: combined Figure 3 and individual panels; both LSTMs use wet-class F1. Older top-level panel filenames remain compatibility copies.
- `predictions/`: canonical neural exports, raw daily sensor labels, and stream orders for reproducible evaluation.
- `map_inputs/`: geographic caches used to reproduce the copula map.

Run `python benchmarks/verify_paper_results.py` to recompute classification from saved predictions and verify the raw copula. Training scripts and data preparation are described in the root README. Copula intervals are conditional simulation intervals; seed standard deviations are training variability, not independent-sample confidence intervals.

Source experiment snapshot: `experiment-archive-2026-09-14`. Full experimental trajectories and local large-artifact checksums remain on `correction-sep11`. The HOBO results were superseded on September 15 by exact three-calendar-day observed targets, using seed 42 only. The HOBO-only LSTM retains 30-observation input histories, uses CPU with two Torch threads, and excludes unlabeled endpoints after sequence construction. Temporal evaluation has 735 targets (584 wet, 151 dry); LR/XGBoost random and site evaluations have 515 and 576 targets. See `baselines/protocol.json`. Augmented-model and copula results are unchanged.
