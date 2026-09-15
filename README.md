# Evaluating Data-Driven Models for Multimodal Prediction of Headwater Streamflow

This branch contains the code and results used in the paper. Exploratory runs, earlier implementations, alternative copulas, and campaign logs are preserved on [`correction-sep11`](https://github.com/michaeltm365/saas-river-forecasting/tree/correction-sep11), with snapshot tag `experiment-archive-2026-09-14`.

## Paper models

| Model | Evaluation | Notebook |
|---|---|---|
| Logistic regression | Random, temporal, and site holdout; causal depth filling | [LR](lr/lr.ipynb) |
| XGBoost | Random, temporal, and site holdout; causal depth filling | [XGBoost](xgb/xgb.ipynb) |
| HOBO-only LSTM | Temporal; seed 42; causal filling and chronological early stopping | [HOBO LSTM](lstm/lstm_hobo_sites.ipynb) |
| LSTM (all sites) | September 10, 2020 cutoff; seeds 42–44; exact calendar t+3 | [All-sites LSTM](lstm/lstm_all_sites.ipynb) |
| RGCN | Same cutoff and seeds; exact calendar t+3; joint discharge prediction | [RGCN](rgcn/rgcn_eval.ipynb) |

The paper is exploratory. All models predict observed water presence exactly three calendar days ahead. Model groups differ in data and split construction. HOBO-only LSTM histories retain 30 observation records; augmented sequence models use calendar-day histories. Random tabular splits describe interpolation. The validation period informed development choices. Seed standard deviations describe training variability on a fixed set, not sampling uncertainty.

The all-sites LSTM and RGCN both include status availability, distinguishing missing status inputs from known dry values, and use dry-only discharge augmentation.

## Results

All published outputs are under [`results/paper/`](results/paper/README.md). Canonical neural classification uses 908 matched sensor targets; full all-sites LSTM scoring uses 956. The HOBO-only LSTM uses 735 temporal targets and achieves accuracy 0.956 and ROC-AUC 0.969. The RGCN sensitivity table contains only the 13 settings discussed in the paper.

On the 908 matched sensor-verified targets, mean ± sample SD over seeds 42–44:

| Model | Accuracy | ROC-AUC | Dry F1 | Dry recall |
|---|---:|---:|---:|---:|
| LSTM (all sites) | 0.951 ± 0.007 | 0.981 ± 0.005 | 0.887 ± 0.016 | 0.919 ± 0.027 |
| RGCN | 0.964 ± 0.003 | 0.986 ± 0.001 | 0.919 ± 0.006 | 0.989 ± 0.005 |

For RGCN seed 42, observed-discharge NSE is 0.757 at Day 1 and 0.480 at Day 3; Day 3 KGE is 0.439. The horizons use slightly different observed target sets.

[Figure 3](results/paper/figure3/figure3.pdf) uses wet-class F1 for both LSTM permutation-importance panels, labeled “permutation Δ F1”. See [figure methods and provenance](results/paper/figure3/README.md).

The copula illustration uses raw RGCN probabilities from seed 42, without probability calibration or a rho cap. It counts dry observations on actual sensor dates, aggregating rolling three-day forecasts retrospectively. Its 95% simulation intervals include the observed count at 18 of 22 reaches. These are observed-period counts, not annual predictions.

## Reproduction

Install the locked environment with `uv sync`. For notebook execution, install a Jupyter frontend separately. Saved canonical prediction snapshots support the RGCN notebook and copula report without retraining or external downloads:

```bash
uv run python benchmarks/paper_canonical_report.py
uv run python benchmarks/verify_paper_results.py
uv run python -m unittest discover -s tests -q
```

To retrain the HOBO models, download inputs with `uv run python download_data.py`, then run:

```bash
uv run python benchmarks/hobo_report.py
```

The HOBO models use seed 42 only. Targets are looked up in the raw sensor observations at issue date + 3 calendar days, with missing targets excluded. LSTM input histories retain rows whose future target is missing.

This writes the LR/XGBoost split results, temporal HOBO LSTM results, predictions, and Figure 3(c). It uses seed 42 and two CPU Torch threads. Required inputs include `obs.csv`, `static_vars.csv`, meteorological drivers, and the graph-derived degree and stream-order tables. ScienceBase downloads may require the manual fallback printed by the downloader. The downloader now retrieves the canonical release, not historical model weights.

For the availability all-sites LSTM:

```bash
uv run python benchmarks/lstm_all_sites.py --prepare
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/lstm_all_sites.py --seed 42
# Repeat for seeds 43 and 44.
```

The runner refuses to overwrite existing artifacts. `uv run python benchmarks/figure3.py` regenerates the combined Figure 3 from saved LSTM importance and verified classical fits. `uv run python benchmarks/lstm_importance.py` recalculates all-sites wet-F1 importance from the saved checkpoint before plotting; use CUDA to reproduce the original numerical settings. See [RGCN reproduction](rgcn/pipeline/README.md) for graph training and sensitivity configurations. Canonical checkpoints and preprocessing are published on Hugging Face. Archived experiment checkpoints remain on the original host. Large training artifacts remain outside Git; the prediction snapshots needed to reproduce the published evaluations are committed here.

Private draft PDFs and editing notes are excluded from this repository branch.


## Canonical model downloads

The [Hugging Face release](https://huggingface.co/michaeltm365/saas-river-forecasting/tree/paper-canonical-2026-09-14) contains the six neural checkpoints (seeds 42–44), exact prepared inputs, graph/node ordering, normalization metadata, configurations, and paper prediction/result snapshots.

```bash
uv run python download_data.py --canonical-only
uv run python benchmarks/verify_paper_results.py
# Optional checkpoint verification on CUDA:
uv run python benchmarks/verify_checkpoints.py
```

The HOBO results and Figure 3 were updated on September 15 using seed 42 and exact calendar-day targets; their current snapshots are committed to GitHub. The Hugging Face tag remains the September 14 release for the unchanged augmented models and preprocessing. The downloader skips superseded HOBO and figure snapshots in that bundle.

The downloader pins `paper-canonical-2026-09-14`, checks SHA-256 for every payload file, and restores the paths expected by the model configurations. It preserves conflicting local files unless `--force` is supplied. The complete release inventory and matching code commit are in `canonical/MANIFEST.json` on Hugging Face. Historical HF weights and predictions remain at tag `historical-before-paper-2026-09-14`.

Raw observations and drivers are attributed to the [USGS data release](https://doi.org/10.5066/P19R5TXW). See the model card for artifact scope and limitations. This release does not introduce a new license for upstream data or software.
