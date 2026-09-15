# Paper files

## Figures

- [Study area](hja_diagram.png)
- [RGCN architecture](rgcn_diagram.png)
- [Figure 3, combined](figure3/figure3.png)
- Figure 3 panels: [LR](figure3/figure3_lr.png), [XGBoost](figure3/figure3_xgb.png), [HOBO LSTM](figure3/figure3_lstm_hobo.png), [All-sites LSTM](figure3/figure3_lstm_all.png)
- [Copula figures](copula/)

`figure3/` contains PNG images only.

## Data

- [Tables](TABLES.md)
- [HOBO results and predictions](baselines/)
- [Neural predictions](predictions/)
- [Feature-importance inputs and provenance](feature_importance/)
- [Sensitivity settings](sensitivity.csv) and [predictions](sensitivity_predictions/)
- [Copula outputs](copula/)

## Regenerate

```bash
uv run python benchmarks/hobo_report.py
uv run python benchmarks/figure3.py
uv run python benchmarks/paper_canonical_report.py
uv run python benchmarks/verify_paper_results.py
```
