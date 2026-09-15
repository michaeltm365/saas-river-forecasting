# SAAS River Forecasting

## Index

- Models: [Logistic Regression](lr/lr.ipynb), [XGBoost](xgb/xgb.ipynb), [HOBO LSTM](lstm/lstm_hobo_sites.ipynb), [All-sites LSTM](lstm/lstm_all_sites.ipynb), [RGCN](rgcn/rgcn_eval.ipynb)
- [Paper figures and results](results/paper/README.md)
- [Model downloads](https://huggingface.co/michaeltm365/saas-river-forecasting/tree/main/canonical)
- [RGCN setup](rgcn/pipeline/README.md)
- [Source data](https://doi.org/10.5066/P19R5TXW)

## Setup

```bash
uv sync
uv run python download_data.py --canonical-only
uv run python benchmarks/verify_paper_results.py
```

For training inputs, run `uv run python download_data.py --include drivers`. Install a Jupyter frontend separately to run notebooks.

## Reproduce

```bash
uv run python benchmarks/hobo_report.py
uv run python benchmarks/figure3.py
uv run python benchmarks/paper_canonical_report.py
# Export model files for publication:
uv run python benchmarks/export_hobo_checkpoints.py
uv run python -m unittest discover -s tests -q
```

[Figure 3](results/paper/figure3/figure3.png) and its individual panels are PNG only. Numerical importance data are in [feature_importance](results/paper/feature_importance/).
