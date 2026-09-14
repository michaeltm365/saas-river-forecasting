"""Training routine for the canonical all-sites LSTM."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hja.data import scale_train_only
from hja.models.lstm import LSTMModel
from hja.evaluation import metrics
OUT = Path("results/paper/lstm_training")
HP = dict(hidden=57, layers=2, dropout=0.15, lr=0.008, batch=64, epochs=15)

def train_eval(split: str, seed: int, dfa: pd.DataFrame, feats: list[str],
               device: torch.device, use_adasyn: bool = False,
               suffix: str = "", extras: dict | None = None) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sequence pool over the raw (unscaled) frame; scaler is fit on the
    # training rows only, then the whole frame is transformed.
    tdates_row = dfa["target_date"]
    if split != "q65":
        raise ValueError("The paper uses only the q65 split")
    row_train = tdates_row <= pd.Timestamp("2020-09-10")
    tr_scaled, all_scaled = scale_train_only(
        dfa.loc[row_train, feats], dfa[feats], feats)
    dfs = dfa.assign(**{c: all_scaled[c].values for c in feats})
    # Released notebook zero-fills AFTER normalization — replicate.
    dfs[feats] = dfs[feats].fillna(0.0)

    X, y, sites, tdates, hobo = make_sequences_dated(dfs, feats)
    grp = np.where(tdates <= pd.Timestamp("2020-09-10"), "train", "val")
    lab_ok = ~np.isnan(y.astype(float))  # mask unlabeled targets (dry_only diet)
    tr, va = (grp == "train") & lab_ok, (grp == "val") & lab_ok
    print(f"[{split} s{seed}] sequences: {tr.sum():,} train / {va.sum():,} val "
          f"/ {(grp == 'drop').sum():,} buffer")

    n, T, d = X[tr].shape
    if use_adasyn:
        raise ValueError("Canonical all-sites training does not resample")
    X_res, y_res = X[tr], y[tr].astype(int)
    idx = np.random.default_rng(seed).permutation(len(X_res))
    cut = int(0.8 * len(idx))
    tr_i, va_i = idx[:cut], idx[cut:]
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    loader = DataLoader(TensorDataset(to_t(X_res[tr_i]),
                                      to_t(y_res[tr_i]).reshape(-1, 1)),
                        batch_size=HP["batch"], shuffle=True)
    Xv = to_t(X_res[va_i]).to(device)
    yv = to_t(y_res[va_i]).reshape(-1, 1).to(device)

    model = LSTMModel(d, HP["hidden"], HP["layers"], HP["dropout"]).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(HP["epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        print(f"  epoch {epoch+1:2d}/{HP['epochs']} val={vl:.4f}")
        if vl < best:
            best, best_state, patience = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    X_va = X[va]
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_va), 4096):
            probs.append(torch.sigmoid(
                model(to_t(X_va[i:i + 4096]).to(device))).cpu().numpy().ravel())
    prob = np.concatenate(probs)
    pred = (prob >= 0.5).astype(int)

    out = pd.DataFrame({
        "site_id": sites[va], "target_date": tdates[va],
        "true_wetdry": y[va].astype(int), "label_is_hobo": hobo[va].astype(int),
        "pred_wetdry_prob": prob,
    })
    out.to_csv(OUT / f"preds_{split}_s{seed}{suffix}.csv", index=False)

    m = {"all": metrics(y[va].astype(int), prob, pred)}
    hb = hobo[va] > 0.5
    if hb.any():
        m["hobo_only"] = metrics(y[va][hb].astype(int), prob[hb], pred[hb])
    if (~hb).any():
        m["discretized_only"] = metrics(y[va][~hb].astype(int), prob[~hb], pred[~hb])
    (OUT / f"metrics_{split}_s{seed}{suffix}.json").write_text(json.dumps(m, indent=2))
    print(f"[{split} s{seed}] all: acc={m['all']['Accuracy']:.3f} "
          f"auc={m['all']['ROC-AUC']:.3f} dryF1={m['all']['DryF1']:.3f} "
          f"(N={m['all']['N']})")
    if extras is not None:
        extras.update(model=model, preds=out, feats=feats,
                      X_val=X_va, y_val=y[va], hobo_val=hobo[va])
    return m
