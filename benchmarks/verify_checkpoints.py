"""Verify downloaded canonical checkpoints against the paper prediction snapshots.

Requires CUDA and the complete bundle from download_data.py --canonical-only.
Does not train models or overwrite the paper snapshots.
"""
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from lstm_all_sites import sequences
from hja.data import BINARY_COLS, scale_train_only
from hja.models.lstm import LSTMModel

ROOT = Path(__file__).resolve().parents[1]

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to match the canonical inference settings.')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    BINARY_COLS.add('status_available')
    for seed in (42, 43, 44):
        ck = torch.load(ROOT/f'data/retrain/paper/lstm_availability_s{seed}.pt', map_location=device, weights_only=False)
        feats, hp = ck['features'], ck['hyperparameters']
        df = pd.read_parquet(ROOT/'data/retrain/paper/lstm_calendar_frame.parquet')
        _, scaled = scale_train_only(df.loc[df.target_date <= '2020-09-10', feats], df[feats], feats)
        for c in feats:
            df[c] = scaled[c].fillna(0).values
        x, y, sites, dates, hobo = sequences(df, feats)
        take = dates > pd.Timestamp('2020-09-10')
        x = x[take]
        model = LSTMModel(len(feats), hp['hidden'], hp['layers'], hp['dropout']).to(device)
        model.load_state_dict(ck['model_state_dict']); model.eval()
        with torch.no_grad():
            p = np.concatenate([torch.sigmoid(model(torch.tensor(x[i:i+4096], device=device))).cpu().numpy().ravel() for i in range(0,len(x),4096)])
        expected = pd.read_csv(ROOT/f'results/paper/predictions/lstm_seed{seed}.csv')
        np.testing.assert_array_equal(sites[take], expected.site_id)
        np.testing.assert_array_equal(dates[take], pd.to_datetime(expected.target_date))
        np.testing.assert_allclose(p, expected.pred_wetdry_prob, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(p >= .5, expected.pred_wetdry_prob >= .5)
        print(f'LSTM seed {seed}: {len(p)} predictions verified', flush=True)
        env = dict(os.environ, RGCN_CONFIG=f'rgcn/configs/seed{seed}.yml')
        subprocess.run([sys.executable, '-m', 'rgcn.pipeline.export_predictions', '--eval-stride', '1', '--day3-range', '2020-09-11:2020-12-31'], cwd=ROOT, env=env, check=True)
        for day in (1,3):
            actual = pd.read_csv(ROOT/f'data/retrain/paper/predictions_rgcn_availability_s{seed}_stride1/train_val_predictions_day{day}.csv').set_index(['site_id','date']).sort_index()
            expected = pd.read_csv(ROOT/f'results/paper/predictions/rgcn_seed{seed}_day{day}.csv').set_index(['site_id','date']).sort_index()
            actual = actual.loc[expected.index]
            for col in ('pred_wetdry_prob','pred_discharge'):
                np.testing.assert_allclose(actual[col], expected[col], atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(actual.pred_wetdry_prob >= .5, expected.pred_wetdry_prob >= .5)
        print(f'RGCN seed {seed}: Day 1 and Day 3 predictions verified', flush=True)
    print('All six canonical checkpoints verified.')

if __name__ == '__main__':
    main()
