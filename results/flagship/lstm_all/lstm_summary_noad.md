# LSTM (all sites) on the flagship splits — NO ADASYN

Optuna released hyperparams; NO resampling (raw class balance, plain BCE); scaler train-only; splits assigned by sequence target date (ph guard=3d). Seeds 42/43/44.

| Split | Scope | Acc (mean ± std) | AUC | Dry F1 | N |
|---|---|--:|--:|--:|--:|
| ph | all | 0.934 ± 0.009 | 0.955 ± 0.006 | 0.770 ± 0.026 | 1060 |
| ph | hobo_only | 0.945 ± 0.013 | 0.961 ± 0.005 | 0.826 ± 0.036 | 638 |
| ph | discretized_only | 0.918 ± 0.003 | 0.949 ± 0.008 | 0.662 ± 0.013 | 422 |
| q65 | all | 0.957 ± 0.004 | 0.938 ± 0.021 | 0.678 ± 0.036 | 10512 |
| q65 | hobo_only | 0.971 ± 0.002 | 0.984 ± 0.002 | 0.929 ± 0.005 | 895 |
| q65 | discretized_only | 0.956 ± 0.004 | 0.932 ± 0.023 | 0.587 ± 0.052 | 9617 |
| q80 | all | 0.955 ± 0.001 | 0.873 ± 0.019 | 0.610 ± 0.005 | 9914 |
| q80 | hobo_only | 0.946 ± 0.003 | 0.968 ± 0.002 | 0.869 ± 0.008 | 490 |
| q80 | discretized_only | 0.955 ± 0.000 | 0.866 ± 0.022 | 0.556 ± 0.006 | 9424 |
| site | all | 0.938 ± 0.041 | 0.970 ± 0.012 | 0.874 ± 0.094 | 449 |
| site | hobo_only | 0.938 ± 0.040 | 0.969 ± 0.013 | 0.873 ± 0.092 | 439 |
| site | discretized_only | 0.933 ± 0.094 | 1.000 ± 0.000 | 0.889 ± 0.157 | 10 |
