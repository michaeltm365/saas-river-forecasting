"""Shared LSTM architecture + sequence construction (used by the HOBO-only
baseline here and by benchmarks/lstm_flagship_splits.py)."""

from __future__ import annotations

import numpy as np
import torch.nn as nn

SEQ_LEN = 30


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_sequences(df, feats, seq_len: int = SEQ_LEN):
    """Per-site sliding windows (label = wet_dry_next at the window's last
    row); also returns each sequence's site id."""
    X, y, sites = [], [], []
    for sid, g in df.groupby("NHDPlusID"):
        if len(g) <= seq_len:
            continue
        f, lab = g[feats].values, g["wet_dry_next"].values
        for i in range(len(g) - seq_len):
            X.append(f[i:i + seq_len])
            y.append(lab[i + seq_len - 1])
            sites.append(sid)
    return np.array(X, dtype=np.float32), np.array(y), np.array(sites)
