# LSTM (all sites) on the flagship splits

Optuna released hyperparams; ADASYN + scaler train-only; splits assigned by sequence target date (ph guard=3d). Seeds 42/43/44.

| Split | Scope | Acc (mean ± std) | AUC | Dry F1 | N |
|---|---|--:|--:|--:|--:|
| ph | all | 0.916 ± 0.003 | 0.964 ± 0.002 | 0.734 ± 0.009 | 1060 |
| ph | hobo_only | 0.916 ± 0.003 | 0.964 ± 0.002 | 0.734 ± 0.009 | 1060 |
| q65 | all | 0.952 ± 0.006 | 0.837 ± 0.037 | 0.659 ± 0.051 | 10512 |
| q65 | hobo_only | 0.952 ± 0.006 | 0.837 ± 0.037 | 0.659 ± 0.051 | 10512 |
| q80 | all | 0.954 ± 0.010 | 0.871 ± 0.076 | 0.654 ± 0.098 | 9914 |
| q80 | hobo_only | 0.954 ± 0.010 | 0.871 ± 0.076 | 0.654 ± 0.098 | 9914 |
| site | all | 0.937 ± 0.035 | 0.975 ± 0.016 | 0.897 ± 0.050 | 449 |
| site | hobo_only | 0.937 ± 0.035 | 0.975 ± 0.016 | 0.897 ± 0.050 | 449 |
