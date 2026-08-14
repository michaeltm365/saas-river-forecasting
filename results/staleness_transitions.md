# Staleness sweep + transition scoring (shared 5-site holdout, seed 42)

Models trained on fresh inputs (site-holdout protocol of results/site_holdout_comparison.md); at evaluation the HELD-OUT site's observation-derived inputs (wet/dry status, MaxDepth) are aged so the latest available observation is k days old. Neighboring sites' observations stay fresh — only the RGCN can exploit them (graph propagation). Persistence = predict the latest available status. Task: wet/dry at t+3.

## Staleness sweep — pooled accuracy (dry recall)

| Model | k=0 | k=3 | k=7 | k=14 | k=21 | k=28 |
|---|--:|--:|--:|--:|--:|--:|
| Persistence | 0.974 (0.92) | 0.966 (0.89) | 0.954 (0.83) | 0.932 (0.73) | 0.908 (0.62) | 0.891 (0.54) |
| LR | 0.828 (0.82) | 0.827 (0.81) | 0.821 (0.79) | 0.810 (0.73) | 0.798 (0.68) | 0.789 (0.64) |
| XGBoost | 0.978 (0.91) | 0.968 (0.87) | 0.954 (0.81) | 0.932 (0.71) | 0.908 (0.60) | 0.891 (0.52) |
| LSTM (HOBO only) | 0.754 (0.52) | 0.754 (0.52) | 0.754 (0.52) | 0.754 (0.52) | 0.754 (0.52) | 0.754 (0.52) |
| LSTM (all sites) | 0.941 (0.99) | 0.943 (1.00) | 0.941 (1.00) | 0.948 (1.00) | 0.945 (1.00) | 0.938 (1.00) |
| RGCN | 0.946 (0.85) | 0.938 (0.84) | 0.922 (0.83) | 0.894 (0.78) | 0.845 (0.67) | 0.803 (0.60) |

N per model: Persistence=588, LR=588, XGBoost=588, LSTM (HOBO only)=443, LSTM (all sites)=439, RGCN=613.

## Transition scoring (fresh inputs, k=0)

Observed status changes per site: {'097170': 5, '099610': 0, '100137': 3, '235848': 0, '271029': 0}. 'Transition' = target dates within ±3 days of a change; 'Stable' = all other labeled dates.

| Model | N trans | Acc (trans) | Dry recall (trans) | N stable | Acc (stable) |
|---|--:|--:|--:|--:|--:|
| Persistence | 33 | 0.545 | 0.600 | 555 | 1.000 |
| LR | 33 | 0.667 | 0.720 | 555 | 0.838 |
| XGBoost | 33 | 0.606 | 0.560 | 555 | 1.000 |
| LSTM (HOBO only) | 31 | 0.677 | 0.720 | 412 | 0.760 |
| LSTM (all sites) | 31 | 0.806 | 0.960 | 408 | 0.951 |
| RGCN | 37 | 0.568 | 0.655 | 576 | 0.970 |

Notes: baselines' target dates use the released notebooks' positional shift(-3) (≈3 calendar days on the near-daily HOBO series); RGCN uses exact calendar t+3. Staleness for baselines is applied per feature row/window; models were not retrained on stale inputs (deployment-mismatch test). All single-seed (42).