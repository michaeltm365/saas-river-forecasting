# LSTM (all sites) on the flagship splits

Optuna released hyperparams; ADASYN + scaler train-only; splits assigned by sequence target date (ph guard=3d). Seeds 42/43/44.

| Split | Scope | Acc (mean ± std) | AUC | Dry F1 | N |
|---|---|--:|--:|--:|--:|
| ph | all | 0.908 ± 0.007 | 0.960 ± 0.004 | 0.718 ± 0.016 | 1060 |
| ph | hobo_only | 0.911 ± 0.009 | 0.969 ± 0.009 | 0.752 ± 0.026 | 638 |
| ph | discretized_only | 0.904 ± 0.009 | 0.945 ± 0.004 | 0.651 ± 0.035 | 422 |
| q65 | all | 0.951 ± 0.008 | 0.870 ± 0.041 | 0.672 ± 0.059 | 10512 |
| q65 | hobo_only | 0.941 ± 0.041 | 0.984 ± 0.000 | 0.876 ± 0.075 | 895 |
| q65 | discretized_only | 0.952 ± 0.008 | 0.851 ± 0.047 | 0.603 ± 0.072 | 9617 |
| q80 | all | 0.944 ± 0.004 | 0.830 ± 0.042 | 0.577 ± 0.013 | 9914 |
| q80 | hobo_only | 0.922 ± 0.028 | 0.968 ± 0.006 | 0.816 ± 0.059 | 490 |
| q80 | discretized_only | 0.946 ± 0.002 | 0.815 ± 0.049 | 0.532 ± 0.005 | 9424 |
| site | all | 0.959 ± 0.014 | 0.982 ± 0.006 | 0.931 ± 0.024 | 449 |
| site | hobo_only | 0.958 ± 0.014 | 0.982 ± 0.007 | 0.928 ± 0.025 | 439 |
| site | discretized_only | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 10 |
