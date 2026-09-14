# LSTM (all sites) on the flagship splits — NO ADASYN

Optuna released hyperparams; NO resampling (raw class balance, plain BCE); scaler train-only; splits assigned by sequence target date (ph guard=3d). Seeds 42/43/44.

| Split | Scope | Acc (mean ± std) | AUC | Dry F1 | N |
|---|---|--:|--:|--:|--:|
| ph | all | 0.924 ± 0.024 | 0.961 ± 0.015 | 0.758 ± 0.053 | 1060 |
| ph | hobo_only | 0.924 ± 0.024 | 0.961 ± 0.015 | 0.758 ± 0.053 | 1060 |
| q65 | all | 0.955 ± 0.001 | 0.908 ± 0.013 | 0.651 ± 0.009 | 10512 |
| q65 | hobo_only | 0.955 ± 0.001 | 0.908 ± 0.013 | 0.651 ± 0.009 | 10512 |
| q80 | all | 0.957 ± 0.005 | 0.893 ± 0.068 | 0.643 ± 0.059 | 9914 |
| q80 | hobo_only | 0.957 ± 0.005 | 0.893 ± 0.068 | 0.643 ± 0.059 | 9914 |
| site | all | 0.914 ± 0.078 | 0.981 ± 0.016 | 0.793 ± 0.215 | 449 |
| site | hobo_only | 0.914 ± 0.078 | 0.981 ± 0.016 | 0.793 ± 0.215 | 449 |
