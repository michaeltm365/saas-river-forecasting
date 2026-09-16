"""Predict one reach/date using the released RGCN and prepared network inputs.

Run from the repository root after download_data.py --canonical-only.
The forecast is evaluated for the entire graph before selecting one reach.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgcn.pipeline import availability, features as F
from rgcn.pipeline.config import load_config
from rgcn.pipeline.data import load_arrays
from rgcn.pipeline.masking import mask_forecast_tail, mask_mode
from rgcn.pipeline.model import create_model
from rgcn.pipeline.windows import build_date_range, WindowSpec


def predict(site_id=55000900097170, target_date='2020-09-20', seed=42, device='cpu'):
    config = load_config(ROOT / f'rgcn/configs/seed{seed}.yml')
    checkpoint = torch.load(config.path('checkpoint'), map_location='cpu', weights_only=False)
    arrays = load_arrays(config)
    dates = build_date_range(config)
    target = pd.Timestamp(target_date)
    spec = WindowSpec.from_config(config)
    end = dates.get_indexer([target])[0] + 1
    start = end - spec.seq_length - spec.forecast_horizon
    if end == 0 or start < 1:
        raise ValueError('Target date must have a complete prepared input window and preceding day')
    nodes = arrays['node_ids'].tolist()
    if site_id not in nodes:
        raise ValueError(f'Reach {site_id} is not in the released graph')
    keep, names = availability.selection(config['features']['exclude_time'], True)
    if names != checkpoint['feature_vars'] or mask_mode(config) != checkpoint['forecast_mask']:
        raise ValueError('Checkpoint features or forecast mask differ from configuration')
    # Include one extra day to construct availability for the first lagged input.
    xt = torch.from_numpy(arrays['X_time'][start-1:end].copy())
    status = torch.from_numpy(arrays['y_all'][start-1:end, :, F.WETDRY_IDX].copy())
    xt = availability.append_channel(xt, status)[1:]
    xt = mask_forecast_tail(xt, spec.seq_length, mask_mode(config), F.N_TIME_FEATURES)
    xt = xt[..., keep]
    static = torch.from_numpy(arrays['X_static']).unsqueeze(0).expand(len(xt), -1, -1)
    x = torch.cat([xt, static], dim=-1).permute(1, 0, 2).to(device)
    # A is stored with the checkpoint and uses its exact node ordering.
    if nodes != checkpoint['node_ids']:
        raise ValueError('Prepared inputs and checkpoint node ordering differ')
    model = create_model(config, checkpoint['model_state_dict']['A'].numpy(), len(names), torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    with torch.inference_mode():
        wet_probability, log_discharge = model(x)[nodes.index(site_id), -1].cpu().tolist()
    return dict(site_id=site_id, seed=seed, issue_date=str((target-pd.Timedelta(days=3)).date()),
                target_date=str(target.date()), input_shape=list(x.shape),
                probability_wet=wet_probability, predicted_status='wet' if wet_probability >= .5 else 'dry',
                discharge_m3_s=float(np.expm1(log_discharge)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-id', type=int, default=55000900097170)
    parser.add_argument('--target-date', default='2020-09-20')
    parser.add_argument('--seed', type=int, choices=[42,43,44], default=42)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(predict(args.site_id, args.target_date, args.seed, args.device), indent=2))


if __name__ == '__main__':
    main()
