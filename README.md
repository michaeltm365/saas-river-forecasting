# Evaluating Data-Driven Models for Multimodal Prediction of Headwater Streamflow

[![USGS](https://img.shields.io/badge/USGS-Collaboration-green)](https://www.usgs.gov/)

## Overview

This repository contains the code and analysis for our research on machine learning approaches for streamflow forecasting in sparse observational networks. Using the H.J. Andrews Forest Long-Term Ecological Research dataset, we evaluate and compare four supervised learning models for predicting wet/dry streamflow status:

- **Logistic Regression** - Baseline classical approach
- **XGBoost** - Gradient-boosted decision trees
- **LSTM** - Long Short-Term Memory neural networks for temporal modeling
- **RGCN** - Recurrent Graph Convolutional Networks for spatial-temporal modeling

## Repository Structure

```
├── src/hja/                         # Shared modeling library (installed with uv sync)
│   ├── data.py                      #   central_df builders, feature selection, scaling
│   ├── splits.py                    #   released splits (random/temporal/site) + flagship constants
│   ├── evaluation.py, importance.py #   shared metrics + feature-importance presentation
│   └── models/                      #   lr / xgb / lstm_hobo runners, each with a CLI
├── lr/
│   └── lr.ipynb                     # Logistic Regression: visualization/demo over hja.models.lr
├── xgb/
│   └── xgb.ipynb                    # XGBoost: visualization/demo over hja.models.xgb
├── lstm/
│   ├── lstm_hobo_sites.ipynb        # LSTM (HOBO sites only): visualization/demo over hja.models.lstm_hobo
│   ├── lstm_all_sites.ipynb         # CANONICAL LSTM (all sites): q65 temporal split, ADASYN
│   └── lstm_all_sites_results.md    # Mixed-data LSTM results (distributional mismatch finding)
├── rgcn/
│   ├── pipeline/                    # Reproducible RGCN training pipeline — see rgcn/pipeline/README.md
│   ├── flagship/                    # CANONICAL RGCN configs (config_q65.yml + split/seed/ablation variants)
│   ├── config.yml                   # Earlier retrain config (+ config_q65.yml / config_phases.yml variants)
│   ├── inspect_driver_weights.py    # Verify drivers/statics are active in a checkpoint
│   ├── build_graph.ipynb            # [as-released] Stream network graph construction
│   ├── train_gnn.ipynb              # [as-released] RGCN model training
│   ├── rgcn_eval.ipynb              # CANONICAL RGCN evaluation: metrics, stream order, annual dry-day copula
│   ├── rgcn_eval_results.md         # [as-released] RGCN evaluation results summary
│   └── rgcn_config.yaml             # [as-released] RGCN model configuration
├── benchmarks/
│   ├── lstm_flagship_splits.py      # CANONICAL LSTM (all sites) runs on the flagship splits
│   ├── flagship_analysis.py         # Matched cross-model comparison + analysis utilities
│   ├── flagship_copula_all.py       # Annual dry-day (Gaussian copula) estimation, all splits
│   └── flagship_ablation_eval.py    # RGCN ablation sweep table
├── synthetic_data/
│   └── gam.ipynb                    # GAM-based synthetic data generation
├── results/
│   ├── flagship/                    # CANONICAL results (FLAGSHIP_RESULTS.md + per-run reports)
│   ├── baselines/                   # LR / XGBoost / LSTM-HOBO metrics from the hja CLIs
│   └── as_released_2026-06/         # Manifest of the archived released baseline (tag: results-as-released)
├── download_data.py                 # Fetch ScienceBase + Hugging Face data into data/
├── classical_lstm_hobo_results.md   # LR, XGBoost, LSTM (HOBO-only) results summary
└── README.md
```

> **Retraining the RGCN:** the released RGCN had three defects (unused
> meteorological drivers, unfed static features, and normalization/split
> leakage). `rgcn/pipeline/` retrains it with the fixes on honest temporal
> splits — full reproduction instructions in
> [`rgcn/pipeline/README.md`](rgcn/pipeline/README.md). The as-released
> baseline is preserved untouched (git tag `results-as-released`,
> `results/as_released_2026-06/MANIFEST.md`).

## Canonical results and how to reproduce them

The paper's canonical neural models are the **flagship RGCN** and the
**LSTM (all sites, ADASYN)**, both evaluated on the **q65 temporal split**
(cutoff 2020-09-10, the 0.65 quantile of wet/dry label dates; training strictly
precedes validation, so no validation-period date appears in any training
input). The flagship RGCN uses 30-day windows, 35 input features (incl. 17
static watershed features, no 7-day lags), and strict forecast-tail masking
(lagged observations, max-depth, AND meteorological drivers frozen at day *t*
for the t+1..t+3 tail — no post-issue-day information). Headline numbers
(seeds 42/43/44, validation, horizons pooled):

| Model | N (val) | Accuracy | ROC-AUC | F1 |
|---|--:|--:|--:|--:|
| RGCN (flagship, q65) | 968 | 0.962 ± 0.009 | 0.984 ± 0.001 | 0.975 ± 0.006 |
| LSTM (all sites, q65) | 10,512 | 0.951 ± 0.008 | 0.870 ± 0.041 | 0.974 ± 0.004 |

The full campaign (all four splits, held-out-site transfer, ablations,
matched cross-model comparison, copula) is consolidated in
[`results/flagship/FLAGSHIP_RESULTS.md`](results/flagship/FLAGSHIP_RESULTS.md).

After setting up the environment and data (sections below), reproduce the
canonical results with:

```bash
# 1) Canonical RGCN (q65, seed 42; use config_q65_s43.yml / _s44.yml for the
#    other seeds — split/array stages are shared and only need the base run)
export RGCN_CONFIG=rgcn/flagship/config_q65.yml
uv run python -m rgcn.pipeline.make_splits
uv run python -m rgcn.pipeline.prepare_data
uv run python -m rgcn.pipeline.build_graph        # once per clone
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.train
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.export_predictions
uv run python -m rgcn.pipeline.eval_report        # -> results/flagship/rgcn_eval_flag_q65.md

# 2) Daily-grid (stride-1) Day-3 export over the val period — needed by the
#    copula section of rgcn/rgcn_eval.ipynb and the analysis scripts
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.export_predictions \
  --eval-stride 1 --day3-range "2020-09-11:2020-12-31"

# 3) Canonical LSTM (all sites): all four flagship splits x seeds 42/43/44
#    (canonical rows = q65) -> results/flagship/lstm_all/
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/lstm_flagship_splits.py

# 4) Matched cross-model tables; copula
uv run python benchmarks/flagship_analysis.py
uv run python benchmarks/flagship_copula_all.py
```

The two canonical notebooks display these results (and are committed with
executed outputs): [`rgcn/rgcn_eval.ipynb`](rgcn/rgcn_eval.ipynb) (RGCN metrics,
stream order, HOBO vs discretized, and the canonical annual dry-day copula
experiment — the q65-trained model on the q65 validation period) and
[`lstm/lstm_all_sites.ipynb`](lstm/lstm_all_sites.ipynb) (canonical LSTM run at
seed 42 plus the multi-seed summary).

## Model Weights

Pre-trained model weights and processed data are available on Hugging Face:

🤗 **[michaeltm365/saas-river-forecasting](https://huggingface.co/michaeltm365/saas-river-forecasting)**

### Quick Download
```python
from huggingface_hub import hf_hub_download

# Download RGCN model weights
model_path = hf_hub_download(
    repo_id="michaeltm365/saas-river-forecasting", 
    filename="best_model.pt"
)

# Download graph structure
graph_path = hf_hub_download(
    repo_id="michaeltm365/saas-river-forecasting", 
    filename="hja_graph.gpickle"
)
```

**Files available:**
- `best_model.pt` - Pre-trained RGCN model weights  
- `hja_graph.gpickle` - H.J. Andrews stream network topology
- `hja_edge_index.npz` - Graph connectivity matrix
- `static_vars_pivot.csv` - Watershed characteristics
- Additional supporting data files

## Data

This project uses data from the H.J. Andrews Forest Long-Term Ecological Research site, including:

- **Observational data**: Continuous discharge measurements and discrete wet/dry classifications
- **Driver variables**: Meteorological data from GridMET (precipitation, temperature, humidity, etc.)
- **Static variables**: Watershed characteristics (slope, elevation, aspect, drainage area)
- **Network topology**: NHDPlus stream segment connectivity

**Note**: Data files are not included in this repository. Please contact the authors or USGS for data access.

## Data Augmentations

1. **Discharge Discretization**: Threshold-based conversion (0.00014 CMS) of continuous measurements to binary wet/dry
2. **Time-Series ADASYN**: Adapted resampling preserving temporal autocorrelation within sliding windows
3. **Synthetic Data via GAMs**: Generalized Additive Models for augmenting sparse observation sites (RMSE=1.33)

## Evaluation Framework

Three train-test splitting strategies for Logistic Regression and XGBoost:

| Strategy | Description | Tests |
|----------|-------------|-------|
| Random | Standard ML benchmarking | General performance |
| Temporal | Chronological split | Forecasting ability |
| Site-based | Entire sites withheld | Spatial generalizability |

## Requirements

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and locked in
`uv.lock` (Python version in `.python-version`). Set up the environment with:

```bash
uv sync
```

On Linux, torch is pinned to the CUDA 12.6 build (runs on CUDA 12.4+ drivers);
other platforms get the default PyPI wheels. RGCN training requires a Linux
CUDA GPU. Fetch data with `uv run python download_data.py` (one 2.13 GB file,
`met_drivers.csv`, requires a manual browser download — the script prints the
URL).

## Usage

All modeling logic lives in the shared `hja` package (`src/hja/`, installed
editable by `uv sync`): data construction (`hja.data`), split strategies
(`hja.splits`), training/evaluation runners (`hja.models`), and shared
metrics/importance presentation (`hja.evaluation`, `hja.importance`). The
notebooks under `lr/`, `xgb/`, and `lstm/` are **visualization/demo layers**
over these functions — the notebooks and the command-line runners execute the
identical code path, so they cannot drift apart.

Run the supervised baselines from the command line (each writes metrics to
`results/baselines/`):

```bash
uv run python -m hja.models.lr         # Logistic Regression, all three released splits
uv run python -m hja.models.xgb        # XGBoost, all three released splits
uv run python -m hja.models.lstm_hobo  # LSTM on HOBO sites (random sequence split)
```

LR and XGBoost reproduce the released notebook numbers exactly (accuracy and
F1 to 6 decimals). The LSTM-HOBO runner fixes a scaler leak in the released
notebook (StandardScaler was fit before the train/test split; it is now fit on
training sequences only), moving random-split accuracy from 0.967 to 0.970.
ROC-AUC is always computed from predicted probabilities.

Example inference (from the notebooks, using a trained runner result `r`):
```python
from hja.models import lr
lr.predict_site_date(r["model"], r["scaler"], frame, r["features"],
                     site_id="HoboSite100", date="2020-10-22")
# "Site HoboSite100 on 2020-10-25 (predicted from 2020-10-22): DRY, (P(wet)=0.0000)"
```

## Citation

If you use this code or findings in your research, please cite:

```
Huang, A., Prieto, C., Murphy, M., Kandadai, A., Wang, A., Yu, A., Krishnan, A.,
Danes, A., Wong, A., Patel, K., Iyer, S., Dubey, V., Nguyen, V., Zwart, J.,
Cook, G., & Chelgren, N. (2025). Evaluating Data-Driven Models for Multimodal Prediction of Headwater Streamflow.
[Preprint in preparation]
```

## Authors

**Student Association for Applied Statistics (SAAS), UC Berkeley**
- Alex Huang, Cristina Prieto, Michael Murphy, Akshath Kandadai, Allison Wang, Amber Yu, Anika Krishnan, Anya Danes, Audrey Wong, Krish Patel, Sanika Iyer, Viksar Dubey, Vivian Nguyen

**United States Geological Survey (USGS)**
- Jacob Zwart, Gericke Cook, Nathan Chelgren

## Acknowledgements

We thank the U.S. Geological Survey for data access and collaboration, and the H.J. Andrews Forest Long-Term Ecological Research program for maintaining the observational network.

---

*For questions or collaboration inquiries, please open an issue or contact the authors.*
