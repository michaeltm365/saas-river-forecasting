"""LSTM baseline on HOBO sensor sites only (random sequence split).

Follows the released lstm_hobo_sites.ipynb protocol (30-day per-site windows,
random 80/20 sequence split, ADASYN on training sequences, hidden 64 / 2
layers / dropout 0.3 / lr 1e-4 / batch 32, early stopping) with ONE deliberate
fix: the released notebook fit its StandardScaler on the WHOLE frame before
splitting, leaking test statistics into scaling. Here the scaler is fit on the
training sequences only. (Released random-split accuracy was 0.967; expect a
slightly different number under the fixed scaler.)

Run:  uv run python -m hja.models.lstm_hobo
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hja.data import build_hobo_frame, feature_frame
from hja.evaluation import metrics, per_class
from hja.models.lstm import LSTMModel, make_sequences
from hja.models.tabular import BASELINES_DIR


def train_eval(frame: pd.DataFrame | None = None, seed: int = 42,
               device: torch.device | None = None,
               hidden: int = 64, layers: int = 2, dropout: float = 0.3,
               lr: float = 1e-4, batch: int = 32, epochs: int = 15,
               patience: int = 5, verbose: bool = True) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The released LSTM-HOBO frame never merges the stream-order table.
    frame = build_hobo_frame(include_order=False) if frame is None else frame
    X_feat, feats = feature_frame(frame)
    df = frame.assign(**{c: X_feat[c].values for c in feats})

    # Sequences from the UNSCALED frame, then split, then fit the scaler on
    # training sequences only (leak fix vs the released global fit).
    X_all, y_all, sites = make_sequences(df, feats)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_all, y_all, sites, test_size=0.2, shuffle=True, random_state=seed)

    n, T, d = X_train.shape
    scaler = StandardScaler().fit(X_train.reshape(-1, d))
    X_train = scaler.transform(X_train.reshape(-1, d)).reshape(n, T, d)
    X_test = scaler.transform(
        X_test.reshape(-1, d)).reshape(len(X_test), T, d)

    # ADASYN on training sequences only (flatten 3D -> 2D, resample, reshape).
    X_res, y_res = ADASYN(random_state=seed).fit_resample(
        X_train.reshape(n, T * d), y_train.astype(int))
    X_res = X_res.reshape(-1, T, d).astype(np.float32)

    # Inner train/val split for early stopping (released protocol).
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_res, y_res, test_size=0.2, random_state=seed)
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
        "sites_test": s_test,
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

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    (BASELINES_DIR / "lstm_hobo_random_split.json").write_text(
        json.dumps({"metrics": m, "per_class": pc}, indent=2))
    (BASELINES_DIR / "lstm_hobo_random_split.md").write_text(
        "# LSTM (HOBO only) — random sequence split\n\n"
        "Released protocol (30-day windows, ADASYN train-only, hidden 64 / 2 "
        "layers / dropout 0.3 / lr 1e-4 / batch 32, early stopping) with the "
        "scaler-leak fix: StandardScaler fit on training sequences only "
        "(the released notebook fit it on the full frame before splitting; "
        "released accuracy was 0.967). Seed 42. ROC-AUC from probabilities.\n\n"
        f"| N | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |\n"
        f"|--:|--:|--:|--:|--:|--:|\n"
        f"| {m['N']} | {m['Accuracy']:.3f} | {m['ROC-AUC']:.3f} | "
        f"{m['WetF1']:.3f} | {m['DryF1']:.3f} | {m['DryRecall']:.3f} |\n\n"
        f"Per-class: dry P/R/F1 {pc['dry']['precision']:.3f}/"
        f"{pc['dry']['recall']:.3f}/{pc['dry']['f1']:.3f} "
        f"(n={pc['dry']['support']}); wet P/R/F1 {pc['wet']['precision']:.3f}/"
        f"{pc['wet']['recall']:.3f}/{pc['wet']['f1']:.3f} "
        f"(n={pc['wet']['support']}).\n")
    print(f"Wrote {BASELINES_DIR / 'lstm_hobo_random_split.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
