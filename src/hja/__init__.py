"""hja — shared modeling library for the SAAS x USGS H.J. Andrews
wet/dry forecasting project.

All data construction, split, training, and evaluation logic for the
supervised baselines (LR / XGBoost / LSTM) lives here; the notebooks under
lr/, xgb/, and lstm/ are visualization/demo layers over these functions, and
the benchmarks/ scripts drive the multi-seed experiment campaigns.
"""
