# xgb — released split protocol

XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1), ADASYN on training rows, no scaling (released protocol; seed 42). ROC-AUC from probabilities.

| Split | N | WetFrac | Accuracy | ROC-AUC | WetF1 | DryF1 | DryRecall |
|---|---|---|---|---|---|---|---|
| random | 526 | 0.838 | 0.987 | 0.999 | 0.992 | 0.959 | 0.965 |
| temporal | 764 | 0.798 | 0.958 | 0.976 | 0.974 | 0.899 | 0.929 |
| site | 581 | 0.644 | 0.905 | 0.958 | 0.924 | 0.875 | 0.932 |

Per-class (dry / wet):

- **random**: dry P/R/F1 0.953/0.965/0.959 (n=85), wet P/R/F1 0.993/0.991/0.992 (n=441)
- **temporal**: dry P/R/F1 0.872/0.929/0.899 (n=154), wet P/R/F1 0.982/0.966/0.974 (n=610)
- **site**: dry P/R/F1 0.825/0.932/0.875 (n=207), wet P/R/F1 0.960/0.890/0.924 (n=374)
