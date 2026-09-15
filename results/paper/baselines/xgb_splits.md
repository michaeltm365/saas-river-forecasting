# xgb — HOBO evaluation protocols with causal depth filling

Exact three-calendar-day observed targets; causal depth filling; seed 42; original split rules.

| Split | N | WetFrac | Accuracy | ROC-AUC | WetF1 | DryF1 | DryRecall |
|---|---|---|---|---|---|---|---|
| random | 515 | 0.827 | 0.983 | 0.999 | 0.989 | 0.950 | 0.966 |
| temporal | 735 | 0.795 | 0.963 | 0.984 | 0.977 | 0.913 | 0.940 |
| site | 576 | 0.644 | 0.967 | 0.980 | 0.975 | 0.953 | 0.937 |

Per-class (dry / wet):

- **random**: dry P/R/F1 0.935/0.966/0.950 (n=89), wet P/R/F1 0.993/0.986/0.989 (n=426)
- **temporal**: dry P/R/F1 0.887/0.940/0.913 (n=151), wet P/R/F1 0.984/0.969/0.977 (n=584)
- **site**: dry P/R/F1 0.970/0.937/0.953 (n=205), wet P/R/F1 0.966/0.984/0.975 (n=371)
