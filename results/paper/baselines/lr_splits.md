# lr — HOBO evaluation protocols with causal depth filling

Exact three-calendar-day observed targets; causal depth filling; seed 42; original split rules.

| Split | N | WetFrac | Accuracy | ROC-AUC | WetF1 | DryF1 | DryRecall |
|---|---|---|---|---|---|---|---|
| random | 515 | 0.827 | 0.983 | 0.992 | 0.989 | 0.952 | 1.000 |
| temporal | 735 | 0.795 | 0.958 | 0.943 | 0.973 | 0.898 | 0.901 |
| site | 576 | 0.644 | 0.802 | 0.892 | 0.865 | 0.632 | 0.478 |

Per-class (dry / wet):

- **random**: dry P/R/F1 0.908/1.000/0.952 (n=89), wet P/R/F1 1.000/0.979/0.989 (n=426)
- **temporal**: dry P/R/F1 0.895/0.901/0.898 (n=151), wet P/R/F1 0.974/0.973/0.973 (n=584)
- **site**: dry P/R/F1 0.933/0.478/0.632 (n=205), wet P/R/F1 0.773/0.981/0.865 (n=371)
