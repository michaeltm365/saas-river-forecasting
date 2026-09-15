"""Export the canonical seed-42 HOBO fits after checking saved predictions.

Run with CPU, two Torch threads, and the project environment. Artifacts are
written under data/retrain/paper/hobo/ for publication on Hugging Face.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from hja.data import build_hobo_frame
from hja.models import lr, xgb, lstm_hobo
from hja.models.lstm import LSTMModel
from hja.paths import REPO


def main():
    torch.set_num_threads(2)
    out = REPO / 'data/retrain/paper/hobo'
    out.mkdir(parents=True, exist_ok=True)
    refs = REPO / 'results/paper/baselines'
    frame = build_hobo_frame()
    frame.to_parquet(out / 'tabular_frame.parquet', index=False)
    for name, module in [('lr', lr), ('xgb', xgb)]:
        for split, result in module.run_all(frame, seed=42).items():
            ref = pd.read_csv(refs / f'{name}_{split}_predictions.csv', dtype={'site':str})
            np.testing.assert_array_equal(result['sites'].astype(str), ref.site)
            np.testing.assert_array_equal(result['y_true'], ref.true_wet)
            np.testing.assert_allclose(result['prob'], ref.prob_wet, rtol=1e-7, atol=1e-8)
            bundle = {key: result[key] for key in ('model', 'scaler', 'features')}
            bundle.update(seed=42, split=split, forecast_horizon_days=3)
            path = out / f'{name}_{split}_seed42.joblib'
            joblib.dump(bundle, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored['model'].predict_proba(
                restored['scaler'].transform(frame[restored['features']])
                if restored['scaler'] is not None else frame[restored['features']]),
                bundle['model'].predict_proba(bundle['scaler'].transform(frame[bundle['features']])
                if bundle['scaler'] is not None else frame[bundle['features']]))
            print('Verified', path.name, flush=True)
    frame = build_hobo_frame(include_order=False, include_target_dates=True, keep_unlabeled=True)
    frame.to_parquet(out / 'lstm_hobo_frame.parquet', index=False)
    result = lstm_hobo.train_eval(frame=frame, seed=42, device=torch.device('cpu'))
    ref = pd.read_csv(refs / 'lstm_hobo_temporal_predictions.csv', parse_dates=['target_date', 'issue_date'])
    np.testing.assert_array_equal(result['sites_test'], ref.site)
    np.testing.assert_array_equal(result['target_dates_test'], ref.target_date)
    np.testing.assert_array_equal(result['y_true'], ref.true_wet)
    np.testing.assert_allclose(result['prob'], ref.prob_wet, atol=1e-8, rtol=1e-7)
    checkpoint = dict(model_state_dict=result['model'].state_dict(), features=result['features'],
                      hyperparameters=dict(hidden=64, layers=2, dropout=.3, lr=.0001,
                                           batch=32, epochs=15, patience=5),
                      seed=42, forecast_horizon_days=3, history_observations=30,
                      split_date=result['split_date'], inner_cutoff=result['inner_cutoff'],
                      scaler_mean=result['scaler'].mean_.tolist(),
                      scaler_scale=result['scaler'].scale_.tolist())
    path = out / 'lstm_hobo_seed42.pt'
    torch.save(checkpoint, path)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    model = LSTMModel(len(restored['features']), 64, 2, .3)
    model.load_state_dict(restored['model_state_dict']); model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.tensor(result['X_test'], dtype=torch.float32))).numpy().ravel()
    np.testing.assert_allclose(probability, ref.prob_wet, rtol=1e-7, atol=1e-8)
    print('Verified', path.name, flush=True)
    (out / 'README.md').write_text('''# HOBO models

- `lr_{random,temporal,site}_seed42.joblib`: estimator, scaler, feature order, split, seed, horizon.
- `xgb_{random,temporal,site}_seed42.joblib`: estimator, feature order, split, seed, horizon.
- `lstm_hobo_seed42.pt`: state dictionary, feature order, architecture, scaling statistics, split, seed, horizon.
- `tabular_frame.parquet`: prepared LR/XGBoost inputs and targets.
- `lstm_hobo_frame.parquet`: prepared LSTM observation rows and calendar targets.

Use the GitHub project's locked environment. Regenerate with `uv run python benchmarks/export_hobo_checkpoints.py`.
''')


if __name__ == '__main__':
    main()
