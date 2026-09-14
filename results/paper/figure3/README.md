# Updated Figure 3

The combined figure and each individual panel are available as vector PDF, vector SVG, and 300-dpi PNG.

- [Combined PDF](figure3.pdf)
- [Combined PNG](figure3.png)
- [Combined SVG](figure3.svg)
- [Logistic regression](figure3_lr.pdf)
- [XGBoost](figure3_xgb.pdf)
- [Temporal HOBO-only LSTM](figure3_lstm_hobo.pdf)
- [All-sites LSTM](figure3_lstm_all.pdf)

The plotting primitives were recovered from the model notebooks at commit `50e33bb` and `src/hja/importance.py`. The two-by-two layout was reconstructed from the Aug 31 figure. None of the old notebook training or importance calculations were reused.

The LR and XGBoost panels use seed-42 temporal fits with causal depth filling. Their probabilities reproduce the canonical saved predictions within CSV precision, and their classifications match exactly. The HOBO-only LSTM panel uses the canonical temporal run (742 targets, seed 42, one permutation per feature, decrease in wet F1). The all-sites panel uses the canonical availability model (956 sensor targets, seed 42, five permutations per feature, decrease in wet F1). The HOBO-only importance values are retained from the canonical wet-F1 calculation at commit `0dd101a`. All-sites importance is recalculated from the canonical checkpoint using wet F1, with unpermuted probabilities verified against the saved canonical predictions. Neither LSTM is retrained.

Signed permutation decreases are retained in the input CSVs. Negative decreases are set to zero before ranking and plotting, rather than converted to positive bars. Equal scores are ordered by feature name. Each panel is scaled by its own largest plotted importance; lengths are not comparable measures of effect across models. `displayed_values.csv` provides every plotted value, and `provenance.json` records source details and input checksums.

From the repository root, with the project environment and raw data available:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python benchmarks/figure3.py
```

To recalculate the all-sites wet-F1 importance from the saved checkpoint before plotting, add `--refresh-lstm-all` to the command above. The HOBO-only panel already uses wet F1.

## Caption

Figure 3. Feature importances across the four data-driven models. Each panel shows the model's top 10 features, with importance max-scaled to 0–1 within each panel and bars colored by feature type. Importance methods differ across models, so values describe relative importance within a panel. (a) Logistic Regression: absolute standardized-coefficient magnitude on the temporal split. (b) XGBoost: gain-based importance on the temporal split. (c) LSTM (HOBO only): permutation Δ F1 after permuting each feature's complete input trajectory across 742 temporal test sequences. (d) LSTM (all sites): permutation Δ F1 across 956 sensor-verified validation targets, averaged over five permutations per feature. F1 refers to the wet class in both LSTM panels. All panels use seed 42. Negative permutation decreases are retained in numerical outputs and set to zero for plotting. Recent wet/dry status denotes the water presence input; its timing follows each model's input construction. Max-scaling does not make effect sizes comparable across panels.
