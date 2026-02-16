# Multimodal Streamflow Forecasting in Sparse Networks

**Addressing Data Scarcity and Spatial Generalization**

[![USGS](https://img.shields.io/badge/USGS-Collaboration-green)](https://www.usgs.gov/)

## Overview

This repository contains the code and analysis for our research on machine learning approaches for streamflow forecasting in sparse observational networks. Using the H.J. Andrews Forest Long-Term Ecological Research dataset, we evaluate and compare four supervised learning models for predicting wet/dry streamflow status:

- **Logistic Regression** - Baseline classical approach
- **XGBoost** - Gradient-boosted decision trees
- **LSTM** - Long Short-Term Memory neural networks for temporal modeling
- **GNN** - Graph Neural Networks for spatial-temporal modeling

Our analysis reveals a fundamental "static feature memorization" problem where classical models achieve high accuracy (0.97-0.99) on standard train-test splits but fail dramatically on site-based evaluations, while Graph Neural Networks maintain ~0.99 accuracy across all evaluation schemes by leveraging spatial topology.

## Repository Structure

```
├── lr/
│   └── lr.ipynb              # Logistic Regression model
├── xgb/
│   └── xgb.ipynb             # XGBoost model
├── lstm/
│   └── lstm.ipynb            # LSTM model
├── gnn/
│   └── train_gnn.ipynb       # Graph Neural Network model
├── synthetic_data/
│   └── gam.ipynb             # GAM-based synthetic data generation
└── README.md
```

## Model Weights

Pre-trained model weights and processed data are available on Hugging Face:

🤗 **[michaeltm365/saas-river-forecasting](https://huggingface.co/michaeltm365/saas-river-forecasting)**

### Quick Download
```python
from huggingface_hub import hf_hub_download

# Download GNN model weights
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
- `best_model.pt` - Pre-trained GNN model weights  
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

## Methodological Innovations

1. **Discharge Discretization**: Threshold-based conversion (0.00014 CMS) of continuous measurements to binary wet/dry
2. **Time-Series ADASYN**: Adapted resampling preserving temporal autocorrelation within sliding windows
3. **Synthetic Data via GAMs**: Generalized Additive Models for augmenting sparse observation sites (RMSE=1.33)
4. **Multi-objective Loss**: Joint optimization of regression and classification tasks for GNNs

## Evaluation Framework

Three train-test splitting strategies for Logistic Regression and XGBoost:

| Strategy | Description | Tests |
|----------|-------------|-------|
| Random | Standard ML benchmarking | General performance |
| Temporal | Chronological split | Forecasting ability |
| Site-based | Entire sites withheld | Spatial generalizability |

## Results Summary

| Model | Random Split | Temporal Split | Site-based Split |
|-------|--------------|----------------|------------------|
| Logistic Regression | 0.97-0.99 | 0.97-0.99 | 0.93 |
| XGBoost | 0.98 | 0.95 | 0.65 |
| LSTM | ~0.90 | ~0.90 | ~0.90 |
| GNN | ~0.99 | ~0.99 | ~0.99 |

## Requirements

```
python>=3.8
pandas
numpy
scikit-learn
xgboost
torch
imbalanced-learn
optuna
matplotlib
seaborn
huggingface_hub
```

For GNN:
```
torch-geometric
```

## Usage

The Logistic Regression, XGBoost, and LSTM notebooks are self-contained and follow a consistent structure:

1. **Imports** - Required libraries
2. **Data Preprocessing** - Loading and merging datasets
3. **Experiments** - Train-test splitting strategies and hyperparameter selection
4. **Model Training** - With class imbalance handling
5. **Evaluation** - Metrics and confusion matrices
6. **Inference** - Function for predicting wet/dry status at new site-date combinations

Example inference:
```python
predict_site_date(
    model=model,
    central_df=central_df,
    site_id="HoboSite100",
    date="2020-10-22"
)
# Output: "Site HoboSite100 forecast for 2020-10-29: WET, (P(wet)=0.8234)"
```

## Citation

If you use this code or findings in your research, please cite:

```
Huang, A., Prieto, C., Murphy, M., Kandadai, A., Wang, A., Yu, A., Krishnan, A.,
Danes, A., Wong, A., Patel, K., Iyer, S., Dubey, V., Nguyen, V., Zwart, J.,
Cook, G., & Chelgren, N. (2025). Multimodal Streamflow Forecasting in Sparse
Networks: Addressing Data Scarcity and Static Feature Memorization.
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
