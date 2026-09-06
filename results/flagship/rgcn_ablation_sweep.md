# Flagship RGCN — loss-function ablations + hyperparameter sensitivity

Flagship protocol: phases split rebuilt with guard_days=3 (784 val wet/dry labels), A-strict masking (obs+drivers tail frozen), no lag-7 features, 30-day windows, seed 42, patience-20 early stopping. One factor changed per row. Single-seed noise on Acc is ~±0.02 (multi-seed flagship: 0.963 ± 0.005); differences inside that band demonstrate robustness, not superiority. Classification pooled over horizons 1-3; discharge in linear CMS. Val loss is not comparable across λ rows.

| Configuration | Best ep | Val loss | Acc | ROC-AUC | F1 | Dry recall | NSE d1 | NSE d3 | KGE d3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| default (λ 1.0/0.5, fpw 2, h64, lr 1e-3, do 0.1, wd 1e-4) | 35 | 0.136 | 0.957 | 0.985 | 0.974 | 0.907 | 0.860 | 0.757 | 0.788 |
| λ_reg 0.5 / λ_cls 1.0 | 49 | 0.196 | 0.964 | 0.985 | 0.978 | 0.922 | 0.819 | 0.754 | 0.747 |
| λ_reg 1.0 / λ_cls 1.0 | 35 | 0.245 | 0.953 | 0.982 | 0.972 | 0.876 | 0.851 | 0.761 | 0.778 |
| discharge-only (λ_cls 0) | 93 | 0.033 | 0.835 | 0.500 | 0.910 | 0.000 | 0.852 | 0.776 | 0.781 |
| wet/dry-only (λ_reg 0) | 17 | 0.116 | 0.959 | 0.981 | 0.975 | 0.922 | -0.277 | -0.269 | -1.060 |
| fpw 1 (unweighted BCE) | 35 | 0.119 | 0.954 | 0.982 | 0.972 | 0.884 | 0.866 | 0.761 | 0.804 |
| fpw 4 | 66 | 0.155 | 0.962 | 0.985 | 0.977 | 0.953 | 0.855 | 0.773 | 0.815 |
| hidden 32 | 35 | 0.151 | 0.953 | 0.978 | 0.971 | 0.946 | 0.864 | 0.751 | 0.788 |
| hidden 128 | 94 | 0.152 | 0.952 | 0.977 | 0.971 | 0.907 | 0.853 | 0.768 | 0.773 |
| lr 3e-4 | 97 | 0.141 | 0.952 | 0.983 | 0.971 | 0.876 | 0.879 | 0.769 | 0.832 |
| lr 3e-3 | 12 | 0.134 | 0.962 | 0.985 | 0.977 | 0.922 | 0.847 | 0.741 | 0.802 |
| dropout 0.0 | 28 | 0.141 | 0.954 | 0.982 | 0.972 | 0.891 | 0.891 | 0.759 | 0.828 |
| dropout 0.3 | 44 | 0.146 | 0.959 | 0.982 | 0.975 | 0.907 | 0.850 | 0.759 | 0.741 |
| weight_decay 0 | 35 | 0.126 | 0.959 | 0.985 | 0.975 | 0.899 | 0.866 | 0.759 | 0.801 |
| weight_decay 1e-3 | 44 | 0.180 | 0.952 | 0.980 | 0.971 | 0.915 | 0.841 | 0.771 | 0.773 |
| no static features | 31 | 0.161 | 0.953 | 0.979 | 0.971 | 0.907 | 0.880 | 0.715 | 0.817 |

Notes: 'wet/dry-only' trains with no discharge loss (its NSE columns test whether the untrained regression head still tracks flow); 'discharge-only' vice versa (classification columns expected near-chance). fpw = dry-class BCE up-weight. 'no static features' drops all 17 NHDPlus watershed vars (input_dim 35 -> 18).