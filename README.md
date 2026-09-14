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
│   ├── lstm_all_sites.ipynb         # CANONICAL LSTM (all sites): q65 split, one-sided labels, no resampling
│   └── lstm_all_sites_results.md    # Mixed-data LSTM results (distributional mismatch finding)
├── rgcn/
│   ├── pipeline/                    # Reproducible RGCN training pipeline — see rgcn/pipeline/README.md
│   ├── flagship/                    # CANONICAL RGCN configs (config_q65.yml + split/seed/ablation variants)
│   ├── config.yml                   # Earlier retrain config (+ config_q65.yml / config_phases.yml variants)
│   ├── inspect_driver_weights.py    # Verify drivers/statics are active in a checkpoint
│   ├── build_graph.ipynb            # [as-released] Stream network graph construction
│   ├── train_gnn.ipynb              # [as-released] RGCN model training
│   ├── rgcn_eval.ipynb              # CANONICAL RGCN evaluation: metrics, stream order, raw seed-42 observed dry counts
│   ├── rgcn_eval_results.md         # [as-released] RGCN evaluation results summary
│   └── rgcn_config.yaml             # [as-released] RGCN model configuration
├── benchmarks/
│   ├── lstm_flagship_splits.py      # CANONICAL LSTM (all sites) runs on the flagship splits
│   ├── flagship_analysis.py         # Persistence baseline + matched cross-model comparison
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

The canonical neural models are now **RGCN + status availability** and
**corrected all-sites LSTM + status availability** (September 13 decision).
Both retain dry-only discharge augmentation and distinguish an unknown
status from a known dry status. The LSTM uses 30 consecutive calendar days,
exact t+3 targets, causal feature filling, no resampling, and cloned best-epoch
weights. The RGCN has 36 inputs (including availability and 17 statics),
no lag-7 features, and strict observation/weather forecast-tail masking.
The availability indicator is lagged and frozen with the status input.

Matched sensor-only q65 validation (cutoff September 10, 2020), exact daily
t+3, N=908, mean ± sample SD over seeds 42/43/44:

| Model | Accuracy | ROC-AUC | Wet F1 | Dry precision | Dry recall |
|---|---:|---:|---:|---:|---:|
| LSTM + availability | 0.951 ± 0.007 | 0.981 ± 0.005 | 0.969 ± 0.005 | 0.858 ± 0.026 | 0.919 ± 0.027 |
| RGCN + availability | 0.964 ± 0.003 | 0.986 ± 0.001 | 0.977 ± 0.002 | 0.858 ± 0.014 | 0.989 ± 0.005 |

The LSTM's full sensor set additionally includes 48 rows outside the RGCN
network (N=956, accuracy 0.954 ± 0.007). Seed SD measures training variability,
not sampling uncertainty. See [canonical tables](results/canonical_availability/TABLES.md),
[raw seed-42 copula outputs](results/canonical_availability/paper/README.md), and
[correction controls](results/correction_sep11/SUMMARY.md).
The old flagship reports are historical. The current main RGCN evaluation is
[`rgcn/rgcn_eval.ipynb`](rgcn/rgcn_eval.ipynb). Paste-ready manuscript changes
are in [the Introduction, Methods, and Results update guide](context/PAPER_UPDATE_INTRO_METHODS_RESULTS.md).

Fresh-clone preparation and seed-42 reproduction (existing training outputs
should be preserved; the corrected LSTM runner refuses to overwrite them):

```bash
# Shared q65 arrays remain the base 20 time features + 17 statics.
# Availability is derived at runtime from unfilled status targets.
export RGCN_CONFIG=rgcn/correction_sep11/config_q65_availability_s42.yml
uv run python -m rgcn.pipeline.make_splits
uv run python -m rgcn.pipeline.prepare_data
uv run python -m rgcn.pipeline.build_graph
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.train
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.export_predictions
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.export_predictions \
  --eval-stride 1 --day3-range "2020-09-11:2020-12-31"

uv run python benchmarks/correction_lstm.py --prepare
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/correction_lstm.py --seed 42 --variant availability
# Repeat with RGCN configs *_s43.yml / *_s44.yml and LSTM --seed 43 / 44.

# After all three seed exports exist:
uv run python benchmarks/availability_products.py  # tables + historical annualization comparison
uv run python benchmarks/observed_period_copula.py  # raw and calibrated sensitivity comparison
uv run python benchmarks/paper_canonical_report.py  # canonical raw seed-42 table and figures
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/availability_products.py --importance
```

The availability-aware 15-variant, seed-42 hyperparameter/ablation sweep is
under `rgcn/availability_sweep/`; [queue status](results/availability_sweep/status.json).
Its classification scorer uses the same 908 sensor-only daily t+3 targets.
The old no-statics/hyperparameter results do not transfer automatically.
The canonical copula uses raw seed-42 probabilities without Platt calibration
or a rho cap. The alternative Platt + rho cap 0.98 implementation remains in
`benchmarks/observed_period_copula.py`. The canonical evaluation uses date-specific
probabilities on exact sensor keys and evaluates
actual dry counts among observed validation dates, preserving calendar gaps.
These aggregate rolling t+3 forecasts retrospectively, rather than predicting
the whole period from one issue date. The earlier 365-day analysis also used
rolling forecasts; its annualized outputs remain historical diagnostics.

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
