"""HOBO-only LSTM with causal depth filling and chronological evaluation.

Thirty-observation histories predict status three observation records ahead.
Test issue dates begin September 15, 2020; training targets precede that date. The last 20% of distinct pre-test
target dates are reserved for early stopping before scaling or ADASYN.
Run: uv run python -m hja.models.lstm_hobo
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hja.data import build_hobo_frame, feature_frame
from hja.evaluation import metrics, per_class
from hja.models.lstm import LSTMModel, make_sequences


def temporal_masks(target_dates, split_date="2020-09-15", issue_dates=None):
    """Split on actual target dates; keep equal dates together across sites."""
    dates = np.asarray(target_dates, dtype="datetime64[ns]")
    cutoff = np.datetime64(split_date)
    earlier = np.unique(dates[dates < cutoff])
    if len(earlier) < 2:
        raise ValueError("Need at least two distinct pre-test target dates")
    inner = earlier[min(max(int(.8 * len(earlier)), 1), len(earlier)-1)]
    masks = dates < inner, (dates >= inner) & (dates < cutoff), dates >= cutoff
    if issue_dates is not None:
        issues = np.asarray(issue_dates, dtype="datetime64[ns]")
        valid = issues < dates
        masks = masks[0] & valid, masks[1] & valid & (issues >= inner), masks[2] & valid & (issues >= cutoff)
    if not all(m.any() for m in masks):
        raise ValueError("Temporal fit, early-stopping, and test sets must be nonempty")
    return *masks, inner


def train_eval(frame: pd.DataFrame | None = None, seed: int = 42,
               device: torch.device | None = None,
               hidden: int = 64, layers: int = 2, dropout: float = 0.3,
               lr: float = 1e-4, batch: int = 32, epochs: int = 15,
               patience: int = 5, verbose: bool = True,
               split_date: str = "2020-09-15") -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The released LSTM-HOBO frame never merges the stream-order table.
    frame = build_hobo_frame(include_order=False, include_target_dates=True) if frame is None else frame
    X_feat, feats = feature_frame(frame, causal=True)
    df = frame.assign(**{c: X_feat[c].values for c in feats})

    if "target_date" not in df:
        raise ValueError("Supply target_date from build_hobo_frame(include_target_dates=True)")
    X_all, y_all, sites = make_sequences(df, feats)
    # Match the shared sequence builder exactly, preserving actual target dates.
    target_dates = np.concatenate([
        g.target_date.to_numpy()[29:len(g)-1]
        for _, g in df.groupby("NHDPlusID") if len(g) > 30
    ])
    issue_dates = np.concatenate([
        g.Date.to_numpy()[29:len(g)-1]
        for _, g in df.groupby("NHDPlusID") if len(g) > 30
    ])
    train_mask, val_mask, test_mask, inner_cutoff = temporal_masks(target_dates, split_date, issue_dates)
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test, s_test = X_all[test_mask], y_all[test_mask], sites[test_mask]
    n, T, d = X_train.shape
    scaler = StandardScaler().fit(X_train.reshape(-1, d))
    def scale_sequences(x):
        return scaler.transform(x.reshape(-1, d)).reshape(x.shape)
    X_train, X_val, X_test = map(scale_sequences, (X_train, X_val, X_test))

    # Synthetic training samples never enter early stopping or testing.
    X_res, y_tr = ADASYN(random_state=seed).fit_resample(
        X_train.reshape(n, T * d), y_train.astype(int))
    X_tr = X_res.reshape(-1, T, d).astype(np.float32)
    if verbose:
        print(f"Target-date split: fit={train_mask.sum()}, early-stop={val_mask.sum()}, "
              f"test={test_mask.sum()}; inner cutoff={inner_cutoff}; test cutoff={split_date}")
    to_t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    train_loader = DataLoader(
        TensorDataset(to_t(X_tr), to_t(y_tr).reshape(-1, 1)),
        batch_size=batch, shuffle=True)
    Xv = to_t(X_val).to(device)
    yv = to_t(y_val).reshape(-1, 1).to(device)

    model = LSTMModel(d, hidden, layers, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb.to(device))
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
            loss = criterion(logits, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            vl = criterion(torch.nan_to_num(model(Xv), nan=0.0, posinf=5.0,
                                            neginf=-5.0), yv).item()
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss={total / len(train_loader):.4f} | Val Loss={vl:.4f}")
        if vl < best:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(to_t(X_test).to(device)).squeeze()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
        prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    y_true = y_test.astype(int)

    return {
        "model": model, "scaler": scaler, "features": feats, "device": device,
        "X_test": X_test, "y_true": y_true, "prob": prob, "pred": pred,
        "sites_test": s_test, "target_dates_test": target_dates[test_mask],
        "issue_dates_test": issue_dates[test_mask],
        "split_date": split_date, "inner_cutoff": str(inner_cutoff),
        "metrics": metrics(y_true, prob, pred),
        "per_class": per_class(y_true, pred),
    }


def permutation_importance(result: dict, seed: int = 42) -> tuple[list, list]:
    """Released permutation importance: shuffle one feature's whole 30-day
    trajectory across test sequences, measure the drop in wet-class F1."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    model, device = result["model"], result["device"]
    X_test, y_true = result["X_test"], result["y_true"]
    base_f1 = f1_score(y_true, result["pred"])

    drops = []
    model.eval()
    for i, _ in enumerate(result["features"]):
        X_perm = X_test.copy()
        X_perm[:, :, i] = X_perm[rng.permutation(len(X_perm))][:, :, i]
        with torch.no_grad():
            logits = model(torch.tensor(X_perm, dtype=torch.float32)
                           .to(device)).squeeze()
            prob = torch.sigmoid(torch.nan_to_num(logits)).cpu().numpy()
        drops.append(base_f1 - f1_score(y_true, (prob >= 0.5).astype(int)))
    return result["features"], drops


def main() -> int:
    res = train_eval()
    m, pc = res["metrics"], res["per_class"]
    print(f"\nAccuracy: {m['Accuracy']:.4f} | ROC-AUC: {m['ROC-AUC']:.4f} | "
          f"Wet F1: {m['WetF1']:.4f} | Dry F1: {m['DryF1']:.4f} (N={m['N']})")

    print(json.dumps({"metrics": m, "per_class": pc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
