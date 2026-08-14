# RGCN loss-function ablations + hyperparameter sensitivity

Phases split (784 val wet/dry labels), consistph protocol (30-day window, obs-masked forecast tail), seed 42, patience-20 early stopping. One factor changed per row from the default. Single-seed noise on Acc is ~±0.02 (see multi-seed replicates); differences inside that band demonstrate robustness, not superiority. Classification pooled over horizons 1-3; discharge in linear CMS. Val loss is not comparable across λ rows.

| Configuration | Best ep | Val loss | Acc | ROC-AUC | F1 | Dry recall | NSE d1 | NSE d3 | KGE d3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| default (λ 1.0/0.5, fpw 2, h64, lr 1e-3, do 0.1, wd 1e-4) | 34 | 0.153 | 0.945 | 0.977 | 0.967 | 0.791 | 0.932 | 0.925 | 0.855 |
| λ_reg 0.5 / λ_cls 1.0 | 17 | 0.272 | 0.948 | 0.977 | 0.968 | 0.876 | 0.915 | 0.863 | 0.905 |
| λ_reg 1.0 / λ_cls 1.0 | 23 | 0.275 | 0.946 | 0.980 | 0.968 | 0.822 | 0.931 | 0.860 | 0.832 |
| discharge-only (λ_cls 0) | 23 | 0.028 | 0.781 | 0.748 | 0.869 | 0.333 | 0.918 | 0.951 | 0.932 |
| wet/dry-only (λ_reg 0) | 8 | 0.152 | 0.948 | 0.966 | 0.968 | 0.876 | -0.718 | -0.758 | -1.085 |
| fpw 1 (unweighted BCE) | 34 | 0.119 | 0.941 | 0.975 | 0.965 | 0.752 | 0.924 | 0.926 | 0.885 |
| fpw 4 | 49 | 0.186 | 0.949 | 0.980 | 0.969 | 0.860 | 0.917 | 0.916 | 0.809 |
| hidden 32 | 50 | 0.152 | 0.949 | 0.972 | 0.970 | 0.837 | 0.874 | 0.897 | 0.828 |
| hidden 128 | 26 | 0.165 | 0.949 | 0.965 | 0.969 | 0.860 | 0.934 | 0.952 | 0.950 |
| lr 3e-4 | 76 | 0.150 | 0.952 | 0.978 | 0.971 | 0.853 | 0.949 | 0.931 | 0.909 |
| lr 3e-3 | 38 | 0.141 | 0.940 | 0.977 | 0.964 | 0.806 | 0.947 | 0.946 | 0.945 |
| dropout 0.0 | 34 | 0.146 | 0.950 | 0.978 | 0.970 | 0.853 | 0.946 | 0.922 | 0.929 |
| dropout 0.3 | 73 | 0.152 | 0.967 | 0.980 | 0.980 | 0.930 | 0.864 | 0.806 | 0.695 |
| weight_decay 0 | 16 | 0.158 | 0.955 | 0.979 | 0.973 | 0.853 | 0.924 | 0.631 | 0.775 |
| weight_decay 1e-3 | 31 | 0.210 | 0.929 | 0.953 | 0.957 | 0.845 | 0.907 | 0.939 | 0.928 |

Notes: 'wet/dry-only' trains with no discharge loss (its NSE columns test whether the untrained regression head still tracks flow); 'discharge-only' vice versa (its classification columns are expected to be near-chance). fpw = dry-class BCE up-weight.