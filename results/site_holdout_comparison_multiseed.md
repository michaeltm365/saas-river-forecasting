# Site-holdout comparison — multi-seed (mean ± std)

Seeds: [42, 43, 44]. Same split/protocol as results/site_holdout_comparison.md.

| Model | N | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 603 | 0.839 ± 0.005 | 0.894 ± 0.000 | 0.890 ± 0.003 | 0.701 ± 0.012 | 0.857 ± 0.022 |
| XGBoost | 603 | 0.976 ± 0.002 | 0.984 ± 0.002 | 0.985 ± 0.001 | 0.944 ± 0.005 | 0.905 ± 0.007 |
| LSTM (HOBO only) | 453 | 0.742 ± 0.015 | 0.890 ± 0.002 | 0.825 ± 0.008 | 0.513 ± 0.043 | 0.471 ± 0.053 |
| LSTM (all sites) | 449 | 0.915 ± 0.033 | 0.971 ± 0.005 | 0.938 ± 0.027 | 0.864 ± 0.040 | 0.912 ± 0.045 |
| RGCN (strict, with-sensor holdout) | 613 | 0.958 ± 0.009 | 0.982 ± 0.002 | 0.973 ± 0.006 | 0.900 ± 0.023 | 0.893 ± 0.032 |