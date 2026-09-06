# LSTM (HOBO only) — random sequence split

Released protocol (30-day windows, ADASYN train-only, hidden 64 / 2 layers / dropout 0.3 / lr 1e-4 / batch 32, early stopping) with the scaler-leak fix: StandardScaler fit on training sequences only (the released notebook fit it on the full frame before splitting; released accuracy was 0.967). Seed 42. ROC-AUC from probabilities.

| N | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |
|--:|--:|--:|--:|--:|--:|
| 394 | 0.970 | 0.991 | 0.981 | 0.925 | 1.000 |

Per-class: dry P/R/F1 0.860/1.000/0.925 (n=74); wet P/R/F1 1.000/0.963/0.981 (n=320).
