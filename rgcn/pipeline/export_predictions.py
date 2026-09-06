"""export_predictions.py — regenerate train_val_predictions_day{1,2,3}.csv (plan §6.5).

Runs the trained RGCN over every window in the split map and writes one CSV per
forecast horizon step (day1/day2/day3 = the 3 future timesteps of each 31-day
window; day-3 = last timestep). Schema is the lean set the eval notebook
(rgcn/rgcn_eval.ipynb) actually consumes, plus discharge:

    window_index, horizon_step, date, site_id, has_true_label,
    true_wetdry, pred_wetdry_prob, pred_wetdry_label,
    true_discharge, pred_discharge      (discharge in linear space)

NOTE: the released files also carried static__*/feature__* columns, which the eval
never reads and would enlarge each file ~30x; they are intentionally omitted.

Run:  CUDA_VISIBLE_DEVICES=2 uv run python -m rgcn.pipeline.export_predictions
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch

from . import features as F
from .config import load_config
from .data import load_arrays
from .masking import mask_forecast_tail, mask_mode
from .model import build_adjacency_matrix, create_model
from .windows import WindowSpec, build_date_range, generate_windows


@torch.no_grad()
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-stride", type=int, default=None,
                    help="Regenerate windows at this stride for inference only "
                         "(e.g. 1 for daily t+3 coverage). Writes to "
                         "<predictions_dir>_stride<N>; window_index then refers "
                         "to the eval grid, NOT the split map.")
    ap.add_argument("--day3-range", default=None, metavar="START:END",
                    help="With --eval-stride: only export windows whose day-3 "
                         "date falls in this range (e.g. 2020-06-01:2020-11-15).")
    args = ap.parse_args()

    config = load_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (launch with CUDA_VISIBLE_DEVICES=...).")
    device = torch.device("cuda:0")

    arr = load_arrays(config)
    X_time = torch.from_numpy(arr["X_time"]).to(device)
    X_static = torch.from_numpy(arr["X_static"]).to(device)
    y_all = arr["y_all"]  # keep on CPU for true-label lookup
    node_ids = arr["node_ids"]
    N = X_static.shape[0]
    static_b = X_static.unsqueeze(0)

    date_range = build_date_range(config)
    spec = WindowSpec.from_config(config)
    if args.eval_stride:
        spec = WindowSpec(spec.seq_length, spec.forecast_horizon, args.eval_stride)
    windows = generate_windows(len(date_range), spec)
    seq_len, horizon = spec.seq_length, spec.forecast_horizon

    if args.eval_stride:
        win_ids = list(range(len(windows)))
        if args.day3_range:
            lo, hi = (pd.Timestamp(x) for x in args.day3_range.split(":"))
            win_ids = [w for w in win_ids
                       if lo <= date_range[windows[w][1] - 1] <= hi]
    else:
        # Export every window in the split map — including "buffer" windows from
        # blocked-holdout splits. They are excluded from train/val metrics by the
        # split filter downstream, but held-out-site scoring (eval_hjflp) needs
        # predictions on those dates too.
        win_ids = sorted(pd.read_csv(config.path("split_map"))["window_index"].tolist())

    graph = pickle.load(open(config.path("graph_out"), "rb"))
    adj = build_adjacency_matrix(graph, node_ids.tolist())

    ckpt = torch.load(config.path("checkpoint"), map_location=device, weights_only=False)
    forecast_mask = mask_mode(config)
    ckpt_mask = ckpt.get("forecast_mask", "none")
    if ckpt_mask != forecast_mask:
        raise RuntimeError(
            f"Config forecast_mask={forecast_mask!r} but checkpoint was trained "
            f"with {ckpt_mask!r} — wrong config/checkpoint pairing."
        )
    exclude_time = (config.get("features") or {}).get("exclude_time") or []
    ckpt_exclude = ckpt.get("exclude_time", [])
    if sorted(ckpt_exclude) != sorted(exclude_time):
        raise RuntimeError(
            f"Config features.exclude_time={exclude_time} but checkpoint was "
            f"trained with {ckpt_exclude} — wrong config/checkpoint pairing."
        )
    exclude_static = bool((config.get("features") or {}).get("exclude_static"))
    ckpt_exclude_static = bool(ckpt.get("exclude_static", False))
    if ckpt_exclude_static != exclude_static:
        raise RuntimeError(
            f"Config features.exclude_static={exclude_static} but checkpoint was "
            f"trained with {ckpt_exclude_static} — wrong config/checkpoint pairing."
        )
    if exclude_static:
        X_static = X_static[:, :0]
        static_b = X_static.unsqueeze(0)
    keep_time_cols, feature_vars = F.time_feature_selection(exclude_time)
    if exclude_static:
        feature_vars = [v for v in feature_vars if v not in F.STATIC_FEATURE_SET]
    input_dim = len(feature_vars)
    keep_idx = (torch.tensor(keep_time_cols, device=device)
                if exclude_time else None)

    model = create_model(config, adj, input_dim, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f}")
    print(f"Exporting {len(win_ids)} windows x {N} sites x {horizon} horizons "
          f"| forecast_mask={forecast_mask}"
          + (f" | exclude_time={exclude_time}" if exclude_time else ""))

    # Per-horizon column buffers.
    buf = {h: {k: [] for k in (
        "window_index", "date", "site_id", "true_wetdry",
        "pred_wetdry_prob", "true_discharge", "pred_discharge",
    )} for h in range(1, horizon + 1)}

    dates_iso = date_range  # DatetimeIndex
    for wid in win_ids:
        start, end = windows[wid]
        # clone: X_time[start:end] is a view and masking writes in place
        xt = mask_forecast_tail(X_time[start:end].clone(), seq_len, forecast_mask)
        if keep_idx is not None:
            xt = xt[..., keep_idx]
        xs = static_b.expand(xt.shape[0], N, X_static.shape[1])
        X = torch.cat([xt, xs], dim=-1).permute(1, 0, 2)   # (N, wl, 37)
        pred = model(X)                                     # (N, wl, 2)
        pred = pred.permute(1, 0, 2).cpu().numpy()          # (wl, N, 2)

        for h in range(1, horizon + 1):
            t_local = seq_len + (h - 1)          # position within window
            t_global = start + t_local
            b = buf[h]
            b["window_index"].append(np.full(N, wid, dtype=np.int64))
            b["date"].append(np.full(N, dates_iso[t_global], dtype="datetime64[ns]"))
            b["site_id"].append(node_ids)
            b["true_wetdry"].append(y_all[t_global, :, F.WETDRY_IDX])
            b["pred_wetdry_prob"].append(pred[t_local, :, F.WETDRY_IDX])
            true_disch_log = y_all[t_global, :, F.DISCHARGE_IDX]
            b["true_discharge"].append(np.expm1(true_disch_log))
            b["pred_discharge"].append(np.expm1(pred[t_local, :, F.DISCHARGE_IDX]))

    out_dir = config.path("predictions_dir")
    if args.eval_stride:
        out_dir = out_dir.with_name(out_dir.name + f"_stride{args.eval_stride}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for h in range(1, horizon + 1):
        b = buf[h]
        df = pd.DataFrame({
            "window_index": np.concatenate(b["window_index"]),
            "horizon_step": h,
            "date": np.concatenate(b["date"]),
            "site_id": np.concatenate(b["site_id"]),
            "true_wetdry": np.concatenate(b["true_wetdry"]),
            "pred_wetdry_prob": np.concatenate(b["pred_wetdry_prob"]),
            "true_discharge": np.concatenate(b["true_discharge"]),
            "pred_discharge": np.concatenate(b["pred_discharge"]),
        })
        df["has_true_label"] = df["true_wetdry"].notna()
        df["pred_wetdry_label"] = (df["pred_wetdry_prob"] >= 0.5).astype(int)
        out = out_dir / f"train_val_predictions_day{h}.csv"
        df.to_csv(out, index=False)
        n_lab = int(df["has_true_label"].sum())
        print(f"  day{h}: {len(df):,} rows ({n_lab:,} with wet/dry label) -> {out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
