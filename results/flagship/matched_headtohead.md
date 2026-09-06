# Matched-set head-to-head at t+3 — flagship RGCN vs LSTM (all sites) vs persistence

All models scored on the identical (reach, date) label rows: the inner join of the RGCN stride-1 day-3 export (val region) and the LSTM's per-sequence predictions, seeds paired 42/43/44 (mean ± std). Persistence uses only the status as of d-3. 'site' = the 5 with-sensor holdout reaches over the full labeled season (RGCN = flag_sh, labels masked from its loss).

## ph  (matched N=651; RGCN val rows 784, LSTM val rows 1060)

| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (flagship) | 651 | 0.958 ± 0.004 | 0.979 ± 0.002 | 0.975 ± 0.002 | 0.879 ± 0.011 | 0.949 ± 0.018 |
| LSTM (ADASYN) | 651 | 0.904 ± 0.010 | 0.957 ± 0.005 | 0.941 ± 0.006 | 0.753 ± 0.023 | 0.905 ± 0.021 |
| LSTM (no ADASYN) | 651 | 0.932 ± 0.015 | 0.949 ± 0.008 | 0.959 ± 0.009 | 0.801 ± 0.038 | 0.841 ± 0.020 |
| Persistence | 651 | 0.945 ± 0.000 | nan ± nan | 0.967 ± 0.000 | 0.824 ± 0.000 | 0.800 ± 0.000 |

HOBO-labeled rows only:

| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (flagship) | 611 | 0.955 ± 0.004 | 0.976 ± 0.002 | 0.973 ± 0.002 | 0.862 ± 0.013 | 0.941 ± 0.021 |
| LSTM (ADASYN) | 611 | 0.907 ± 0.010 | 0.969 ± 0.008 | 0.943 ± 0.006 | 0.752 ± 0.026 | 0.945 ± 0.036 |
| LSTM (no ADASYN) | 611 | 0.943 ± 0.014 | 0.960 ± 0.004 | 0.966 ± 0.009 | 0.826 ± 0.036 | 0.905 ± 0.021 |
| Persistence | 611 | 0.953 ± 0.000 | nan ± nan | 0.972 ± 0.000 | 0.842 ± 0.000 | 0.846 ± 0.000 |

## q65  (matched N=928; RGCN val rows 945, LSTM val rows 10512)

| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (flagship) | 928 | 0.961 ± 0.010 | 0.982 ± 0.001 | 0.974 ± 0.006 | 0.921 ± 0.023 | 0.946 ± 0.048 |
| LSTM (ADASYN) | 928 | 0.926 ± 0.037 | 0.967 ± 0.002 | 0.949 ± 0.027 | 0.860 ± 0.057 | 0.915 ± 0.018 |
| LSTM (no ADASYN) | 928 | 0.941 ± 0.003 | 0.963 ± 0.000 | 0.962 ± 0.002 | 0.873 ± 0.007 | 0.839 ± 0.013 |
| Persistence | 928 | 0.946 ± 0.000 | nan ± nan | 0.965 ± 0.000 | 0.884 ± 0.000 | 0.852 ± 0.000 |

HOBO-labeled rows only:

| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (flagship) | 851 | 0.962 ± 0.009 | 0.986 ± 0.001 | 0.976 ± 0.006 | 0.913 ± 0.024 | 0.955 ± 0.052 |
| LSTM (ADASYN) | 851 | 0.938 ± 0.043 | 0.984 ± 0.001 | 0.958 ± 0.030 | 0.876 ± 0.075 | 0.983 ± 0.008 |
| LSTM (no ADASYN) | 851 | 0.969 ± 0.002 | 0.984 ± 0.002 | 0.981 ± 0.001 | 0.929 ± 0.005 | 0.957 ± 0.010 |
| Persistence | 851 | 0.967 ± 0.000 | nan ± nan | 0.979 ± 0.000 | 0.923 ± 0.000 | 0.938 ± 0.000 |

## q80  (matched N=537; RGCN val rows 554, LSTM val rows 9914)

| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (flagship) | 537 | 0.942 ± 0.005 | 0.969 ± 0.002 | 0.961 ± 0.003 | 0.888 ± 0.011 | 0.944 ± 0.026 |
| LSTM (ADASYN) | 537 | 0.919 ± 0.005 | 0.946 ± 0.006 | 0.947 ± 0.003 | 0.835 ± 0.009 | 0.846 ± 0.019 |
| LSTM (no ADASYN) | 537 | 0.910 ± 0.005 | 0.948 ± 0.000 | 0.940 ± 0.003 | 0.816 ± 0.013 | 0.826 ± 0.025 |
| Persistence | 537 | 0.914 ± 0.000 | nan ± nan | 0.944 ± 0.000 | 0.815 ± 0.000 | 0.777 ± 0.000 |

HOBO-labeled rows only:

| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (flagship) | 464 | 0.941 ± 0.004 | 0.975 ± 0.003 | 0.963 ± 0.002 | 0.862 ± 0.011 | 0.959 ± 0.028 |
| LSTM (ADASYN) | 464 | 0.938 ± 0.006 | 0.969 ± 0.004 | 0.960 ± 0.004 | 0.852 ± 0.016 | 0.936 ± 0.037 |
| LSTM (no ADASYN) | 464 | 0.943 ± 0.004 | 0.966 ± 0.003 | 0.964 ± 0.002 | 0.869 ± 0.008 | 0.985 ± 0.005 |
| Persistence | 464 | 0.940 ± 0.000 | nan ± nan | 0.962 ± 0.000 | 0.848 ± 0.000 | 0.876 ± 0.000 |

## site  (matched N=449; RGCN val rows 613, LSTM val rows 449)

| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (flagship) | 449 | 0.928 ± 0.034 | 0.970 ± 0.013 | 0.949 ± 0.025 | 0.878 ± 0.055 | 0.889 ± 0.044 |
| LSTM (ADASYN) | 449 | 0.959 ± 0.014 | 0.982 ± 0.006 | 0.971 ± 0.010 | 0.931 ± 0.024 | 0.956 ± 0.035 |
| LSTM (no ADASYN) | 449 | 0.938 ± 0.041 | 0.970 ± 0.012 | 0.958 ± 0.026 | 0.874 ± 0.094 | 0.809 ± 0.150 |
| Persistence | 449 | 0.964 ± 0.000 | nan ± nan | 0.975 ± 0.000 | 0.937 ± 0.000 | 0.915 ± 0.000 |

HOBO-labeled rows only:

| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (flagship) | 439 | 0.927 ± 0.035 | 0.970 ± 0.014 | 0.948 ± 0.026 | 0.875 ± 0.058 | 0.888 ± 0.046 |
| LSTM (ADASYN) | 439 | 0.958 ± 0.014 | 0.982 ± 0.007 | 0.971 ± 0.010 | 0.928 ± 0.025 | 0.955 ± 0.036 |
| LSTM (no ADASYN) | 439 | 0.938 ± 0.040 | 0.969 ± 0.013 | 0.959 ± 0.025 | 0.873 ± 0.092 | 0.808 ± 0.148 |
| Persistence | 439 | 0.966 ± 0.000 | nan ± nan | 0.976 ± 0.000 | 0.939 ± 0.000 | 0.920 ± 0.000 |

