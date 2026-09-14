# Evaluating Data-Driven Models for Multimodal Prediction of Headwater Streamflow

This branch contains the code and results used in the paper. Exploratory runs, earlier implementations, alternative copulas, and campaign logs are preserved on [`correction-sep11`](https://github.com/michaeltm365/saas-river-forecasting/tree/correction-sep11), with snapshot tag `experiment-archive-2026-09-14`.

## Paper models

| Model | Evaluation | Notebook |
|---|---|---|
| Logistic regression | Random, temporal, and site holdout; causal depth filling | [LR](lr/lr.ipynb) |
| XGBoost | Random, temporal, and site holdout; causal depth filling | [XGBoost](xgb/xgb.ipynb) |
| HOBO-only LSTM | Temporal; seed 42; causal filling and chronological early stopping | [HOBO LSTM](lstm/lstm_hobo_sites.ipynb) |
| All-sites LSTM + availability | September 10, 2020 cutoff; seeds 42–44; exact calendar t+3 | [All-sites LSTM](lstm/lstm_all_sites.ipynb) |
| RGCN + availability | Same cutoff and seeds; exact calendar t+3; joint discharge prediction | [RGCN](rgcn/rgcn_eval.ipynb) |

The paper is exploratory. Model groups differ in data, splits, and target construction. HOBO-only models retain targets three observation records ahead; the augmented models use three calendar days. Random tabular splits describe interpolation. The validation period informed development choices. Seed standard deviations describe training variability on a fixed set, not sampling uncertainty.

## Results

All published outputs are under [`results/paper/`](results/paper/README.md). Canonical neural classification uses 908 matched sensor targets; full all-sites LSTM scoring uses 956. The HOBO-only LSTM uses 742 temporal targets and achieves accuracy 0.950 and ROC-AUC 0.966. The RGCN sensitivity table contains only the 13 settings discussed in the paper.

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

This writes the LR/XGBoost split results, temporal HOBO LSTM results, predictions, and Figure 3(c). It uses seed 42 and two CPU Torch threads. Required inputs include `obs.csv`, `static_vars.csv`, meteorological drivers, and the graph-derived degree and stream-order tables. ScienceBase downloads may require the manual fallback printed by the downloader. Downloaded historical model weights are not the canonical trained models; use the training commands here.

For the availability all-sites LSTM:

```bash
uv run python benchmarks/lstm_all_sites.py --prepare
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/lstm_all_sites.py --seed 42
# Repeat for seeds 43 and 44.
```

The runner refuses to overwrite existing artifacts. After training seed 42, `uv run python benchmarks/lstm_importance.py` regenerates Figure 3(d). See [RGCN reproduction](rgcn/pipeline/README.md) for graph training and sensitivity configurations. Archived checkpoint checksums and host-storage locations are documented on the experiment branch. Large training artifacts remain outside Git; the prediction snapshots needed to reproduce the published evaluations are committed here.

Private draft PDFs and editing notes are excluded from this repository branch.
