# Flagship RGCN — loss-function ablations + hyperparameter sensitivity

Flagship protocol: q65 temporal split, cutoff 2020-09-10; classification at the Day-3 horizon only (canonical convention; 314 stride-3 val labels), A-strict masking (obs+drivers tail frozen), no lag-7 features, 30-day windows, seed 42, patience-20 early stopping. One factor changed per row. Single-seed noise on Acc is ~±0.02-0.03 at N=314 (multi-seed flagship q65 Day-3: 0.962 ± 0.013); differences inside that band demonstrate robustness, not superiority. Discharge in linear CMS. Val loss is not comparable across λ rows.

| Configuration | Best ep | Val loss | Acc | ROC-AUC | F1 | Dry recall | NSE d1 | NSE d3 | KGE d3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| default (λ 1.0/0.5, fpw 2, h64, lr 1e-3, do 0.1, wd 1e-4) | 42 | 0.147 | 0.943 | 0.982 | 0.962 | 0.870 | 0.938 | 0.396 | 0.392 |
| λ_reg 0.5 / λ_cls 1.0 | 53 | 0.156 | 0.965 | 0.982 | 0.977 | 0.948 | 0.918 | 0.396 | 0.396 |
| λ_reg 1.0 / λ_cls 1.0 | 41 | 0.202 | 0.962 | 0.982 | 0.974 | 0.948 | 0.910 | 0.408 | 0.354 |
| discharge-only (λ_cls 0) | 85 | 0.075 | 0.755 | 0.500 | 0.860 | 0.000 | 0.939 | 0.418 | 0.301 |
| wet/dry-only (λ_reg 0) | 21 | 0.062 | 0.965 | 0.984 | 0.976 | 0.974 | -0.047 | -0.042 | -0.903 |
| fpw 1 (unweighted BCE) | 31 | 0.124 | 0.946 | 0.983 | 0.964 | 0.870 | 0.906 | 0.416 | 0.366 |
| fpw 4 | 42 | 0.178 | 0.962 | 0.981 | 0.974 | 0.948 | 0.938 | 0.398 | 0.404 |
| hidden 32 | 55 | 0.153 | 0.943 | 0.977 | 0.962 | 0.896 | 0.883 | 0.398 | 0.406 |
| hidden 128 | 59 | 0.134 | 0.965 | 0.982 | 0.976 | 0.987 | 0.859 | 0.404 | 0.263 |
| lr 3e-4 | 98 | 0.137 | 0.968 | 0.982 | 0.979 | 0.974 | 0.914 | 0.405 | 0.365 |
| lr 3e-3 | 45 | 0.136 | 0.962 | 0.980 | 0.974 | 0.948 | 0.953 | 0.389 | 0.409 |
| dropout 0.0 | 90 | 0.123 | 0.968 | 0.987 | 0.979 | 0.987 | 0.932 | 0.464 | 0.366 |
| dropout 0.3 | 71 | 0.165 | 0.965 | 0.983 | 0.977 | 0.948 | 0.784 | 0.344 | 0.188 |
| weight_decay 0 | 26 | 0.158 | 0.939 | 0.978 | 0.960 | 0.844 | 0.899 | 0.405 | 0.376 |
| weight_decay 1e-3 | 55 | 0.154 | 0.962 | 0.984 | 0.974 | 0.974 | 0.865 | 0.385 | 0.386 |
| no static features | 25 | 0.146 | 0.968 | 0.979 | 0.979 | 0.987 | 0.877 | 0.379 | 0.423 |

Notes: 'wet/dry-only' trains with no discharge loss (its NSE columns test whether the untrained regression head still tracks flow); 'discharge-only' vice versa (classification columns expected near-chance). fpw = dry-class BCE up-weight. 'no static features' drops all 17 NHDPlus watershed vars (input_dim 35 -> 18).