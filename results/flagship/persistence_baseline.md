# Persistence baseline on the flagship splits

Horizon-matched persistence: the day-h forecast for date d is the last observed status (HOBO preferred, else discretized discharge) as of d-h, forward-filled. Scored on the exact rows of the flagship RGCN's stride-3 val exports (has_true_label & val dates). RGCN rows are seed 42; flagship multi-seed means are in the eval reports.

## ph

| Model / horizon | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Persistence d1 | 263 | 0.970 | nan | 0.982 | 0.907 | 0.867 |
| RGCN d1 (s42) | 263 | 0.951 | 0.986 | 0.970 | 0.857 | 0.867 |
| Persistence d2 | 264 | 0.970 | nan | 0.982 | 0.902 | 0.902 |
| RGCN d2 (s42) | 264 | 0.958 | 0.982 | 0.975 | 0.874 | 0.927 |
| Persistence d3 | 257 | 0.961 | nan | 0.977 | 0.881 | 0.860 |
| RGCN d3 (s42) | 257 | 0.961 | 0.985 | 0.976 | 0.889 | 0.930 |
| Persistence pooled | 784 | 0.967 | nan | 0.980 | 0.897 | 0.876 |
| RGCN pooled (s42) | 784 | 0.957 | 0.985 | 0.974 | 0.873 | 0.907 |

## q65

| Model / horizon | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Persistence d1 | 309 | 0.964 | nan | 0.977 | 0.924 | 0.905 |
| RGCN d1 (s42) | 309 | 0.955 | 0.984 | 0.970 | 0.905 | 0.905 |
| Persistence d2 | 322 | 0.972 | nan | 0.982 | 0.940 | 0.933 |
| RGCN d2 (s42) | 322 | 0.950 | 0.980 | 0.968 | 0.893 | 0.893 |
| Persistence d3 | 314 | 0.949 | nan | 0.967 | 0.892 | 0.857 |
| RGCN d3 (s42) | 314 | 0.943 | 0.982 | 0.962 | 0.882 | 0.870 |
| Persistence pooled | 945 | 0.962 | nan | 0.975 | 0.919 | 0.898 |
| RGCN pooled (s42) | 945 | 0.949 | 0.982 | 0.967 | 0.893 | 0.889 |

## q80

| Model / horizon | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Persistence d1 | 178 | 0.944 | nan | 0.963 | 0.881 | 0.860 |
| RGCN d1 (s42) | 178 | 0.944 | 0.970 | 0.962 | 0.891 | 0.953 |
| Persistence d2 | 192 | 0.953 | nan | 0.970 | 0.897 | 0.886 |
| RGCN d2 (s42) | 192 | 0.953 | 0.970 | 0.969 | 0.903 | 0.955 |
| Persistence d3 | 184 | 0.918 | nan | 0.947 | 0.828 | 0.783 |
| RGCN d3 (s42) | 184 | 0.935 | 0.972 | 0.956 | 0.875 | 0.913 |
| Persistence pooled | 554 | 0.939 | nan | 0.960 | 0.868 | 0.842 |
| RGCN pooled (s42) | 554 | 0.944 | 0.971 | 0.963 | 0.890 | 0.940 |

