# Paper results

- `TABLES.md`, `classification_seeds.csv`: availability-model classification by seed, full/matched sensor set, and stream order.
- `baselines/`: causal LR/XGBoost results and temporal HOBO-only LSTM results, with prediction snapshots.
- `sensitivity.csv`: the default and twelve manuscript sensitivity settings, seed 42. `sensitivity_predictions/` contains the sensor and observed-discharge predictions supporting every row.
- `copula/`: raw seed-42 observed-period counts, intervals, and maps.
- `figure3c_lstm_hobo.*`, `figure3d_lstm_availability.*`: current LSTM importance panels.
- `predictions/`: canonical neural exports, raw daily sensor labels, and stream orders for reproducible evaluation.
- `map_inputs/`: geographic caches used to reproduce the copula map.

Run `python benchmarks/verify_paper_results.py` to recompute classification from saved predictions and verify the raw copula. Training scripts and data preparation are described in the root README. Copula intervals are conditional simulation intervals; seed standard deviations are training variability, not independent-sample confidence intervals.

Source experiment snapshot: `experiment-archive-2026-09-14`. Full experimental trajectories and local large-artifact checksums remain on `correction-sep11`. The new temporal HOBO-only outputs were regenerated from the same seed-42 CPU protocol during this cleanup.
