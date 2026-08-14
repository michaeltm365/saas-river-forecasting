"""Fast in-memory eval of RGCN ablation checkpoints on the phases val split.

Loads each ablation config + checkpoint, runs only the val windows (no CSV
export), and assembles one Table-8-style report: wet/dry Acc / ROC-AUC / F1 /
dry recall (pooled over horizons 1-3) + discharge NSE & KGE (day-1, day-3,
linear CMS). Includes the default consistph checkpoint as the center row.

Run:  CUDA_VISIBLE_DEVICES=<n> uv run python benchmarks/rgcn_ablation_eval.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rgcn.pipeline import features as F  # noqa: E402
from rgcn.pipeline.config import load_config  # noqa: E402
from rgcn.pipeline.data import load_arrays  # noqa: E402
from rgcn.pipeline.dataset import load_split_indices  # noqa: E402
from rgcn.pipeline.masking import mask_forecast_tail, mask_mode  # noqa: E402
from rgcn.pipeline.model import build_adjacency_matrix, create_model  # noqa: E402
from rgcn.pipeline.windows import (  # noqa: E402
    WindowSpec, build_date_range, generate_windows)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROWS = [
    # (label, config path or None for the default consistph center row)
    ("default (λ 1.0/0.5, fpw 2, h64, lr 1e-3, do 0.1, wd 1e-4)", None),
    ("λ_reg 0.5 / λ_cls 1.0", "rgcn/ablations/config_l05c10.yml"),
    ("λ_reg 1.0 / λ_cls 1.0", "rgcn/ablations/config_l10c10.yml"),
    ("discharge-only (λ_cls 0)", "rgcn/ablations/config_reg_only.yml"),
    ("wet/dry-only (λ_reg 0)", "rgcn/ablations/config_cls_only.yml"),
    ("fpw 1 (unweighted BCE)", "rgcn/ablations/config_fpw1.yml"),
    ("fpw 4", "rgcn/ablations/config_fpw4.yml"),
    ("hidden 32", "rgcn/ablations/config_h32.yml"),
    ("hidden 128", "rgcn/ablations/config_h128.yml"),
    ("lr 3e-4", "rgcn/ablations/config_lr3e4.yml"),
    ("lr 3e-3", "rgcn/ablations/config_lr3e3.yml"),
    ("dropout 0.0", "rgcn/ablations/config_do00.yml"),
    ("dropout 0.3", "rgcn/ablations/config_do03.yml"),
    ("weight_decay 0", "rgcn/ablations/config_wd0.yml"),
    ("weight_decay 1e-3", "rgcn/ablations/config_wd1e3.yml"),
]


def nse_kge(t, p):
    m = ~np.isnan(t) & ~np.isnan(p)
    t, p = t[m], p[m]
    if len(t) < 2:
        return float("nan"), float("nan")
    nse = 1 - np.sum((t - p) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-10)
    r = np.corrcoef(t, p)[0, 1]
    kge = 1 - np.sqrt((r - 1) ** 2 + (p.std() / (t.std() + 1e-10) - 1) ** 2
                      + (p.mean() / (t.mean() + 1e-10) - 1) ** 2)
    return float(nse), float(kge)


@torch.no_grad()
def eval_ckpt(config_path):
    config = load_config(REPO / (config_path or "rgcn/config_consistph.yml"))
    arr = load_arrays(config)
    X_time = torch.from_numpy(arr["X_time"]).to(DEVICE)
    X_static = torch.from_numpy(arr["X_static"]).to(DEVICE)
    y_all = arr["y_all"]
    node_ids = arr["node_ids"].tolist()

    spec = WindowSpec.from_config(config)
    date_range = build_date_range(config)
    windows = generate_windows(len(date_range), spec)
    val_ids = load_split_indices(config.path("split_map"))["val"]
    seq_len, horizon = spec.seq_length, spec.forecast_horizon
    fmask = mask_mode(config)

    ckpt = torch.load(config.path("checkpoint"), map_location=DEVICE,
                      weights_only=False)
    keep_cols, feature_vars = F.time_feature_selection(ckpt.get("exclude_time", []))
    model = create_model(config, adj_matrix=build_adjacency_matrix(
        pickle.load(open(config.path("graph_out"), "rb")), node_ids),
        input_dim=len(feature_vars), device=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    keep_idx = torch.tensor(keep_cols, device=DEVICE)

    cls_y, cls_p = [], []
    disch = {1: ([], []), 3: ([], [])}
    for wid in val_ids:
        start, end = windows[wid]
        xt = mask_forecast_tail(X_time[start:end].clone()[None], seq_len, fmask)
        xt = xt[..., keep_idx]
        xs = X_static[None, None].expand(1, xt.shape[1], *X_static.shape)
        pred = model(torch.cat([xt, xs], dim=-1).permute(0, 2, 1, 3))[0]  # (N, wl, 2)
        pred = pred.cpu().numpy()
        for h in range(1, horizon + 1):
            tl = seq_len + h - 1
            tg = start + tl
            yw = y_all[tg, :, F.WETDRY_IDX]
            ok = ~np.isnan(yw)
            cls_y.append(yw[ok].round())
            cls_p.append(pred[ok, tl, F.WETDRY_IDX])
            if h in disch:
                yd = y_all[tg, :, F.DISCHARGE_IDX]
                okd = ~np.isnan(yd)
                disch[h][0].append(np.expm1(yd[okd]))
                disch[h][1].append(np.expm1(pred[okd, tl, F.DISCHARGE_IDX]))

    y = np.concatenate(cls_y).astype(int)
    p = np.concatenate(cls_p)
    lab = (p >= 0.5).astype(int)
    row = {
        "epoch": int(ckpt.get("epoch", -1)) + 1,
        "val_loss": float(ckpt.get("val_loss", float("nan"))),
        "N": len(y),
        "Acc": accuracy_score(y, lab),
        "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        "F1": f1_score(y, lab, zero_division=0),
        "DryRec": (float(((y == 0) & (lab == 0)).sum() / max((y == 0).sum(), 1))),
    }
    for h in (1, 3):
        t = np.concatenate(disch[h][0])
        pr = np.concatenate(disch[h][1])
        row[f"NSE{h}"], row[f"KGE{h}"] = nse_kge(t, pr)
    return row


def main() -> int:
    lines = ["# RGCN loss-function ablations + hyperparameter sensitivity", ""]
    lines.append("Phases split (784 val wet/dry labels), consistph protocol "
                 "(30-day window, obs-masked forecast tail), seed 42, patience-20 "
                 "early stopping. One factor changed per row from the default. "
                 "Single-seed noise on Acc is ~±0.02 (see multi-seed replicates); "
                 "differences inside that band demonstrate robustness, not "
                 "superiority. Classification pooled over horizons 1-3; discharge "
                 "in linear CMS. Val loss is not comparable across λ rows.")
    lines.append("")
    lines.append("| Configuration | Best ep | Val loss | Acc | ROC-AUC | F1 | "
                 "Dry recall | NSE d1 | NSE d3 | KGE d3 |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for label, cfg in ROWS:
        try:
            r = eval_ckpt(cfg)
        except FileNotFoundError as e:
            print(f"SKIP {label}: {e}")
            continue
        lines.append(f"| {label} | {r['epoch']} | {r['val_loss']:.3f} | "
                     f"{r['Acc']:.3f} | {r['AUC']:.3f} | {r['F1']:.3f} | "
                     f"{r['DryRec']:.3f} | {r['NSE1']:.3f} | {r['NSE3']:.3f} | "
                     f"{r['KGE3']:.3f} |")
        print(f"done: {label}", flush=True)
    lines.append("")
    lines.append("Notes: 'wet/dry-only' trains with no discharge loss (its NSE "
                 "columns test whether the untrained regression head still tracks "
                 "flow); 'discharge-only' vice versa (its classification columns "
                 "are expected to be near-chance). fpw = dry-class BCE up-weight.")
    out = REPO / "results/rgcn_ablation_sweep.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
