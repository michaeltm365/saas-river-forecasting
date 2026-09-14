> **Historical annualization diagnostic.** The current observed-period evaluation is in [OBSERVED_COPULA.md](OBSERVED_COPULA.md). This 365-day setup also used rolling t+3 forecasts, not a single forecast issued at the beginning of the period.

# Availability-RGCN copula comparison

Both setups use identical exact sensor keys and 10,000 simulations/site. Current setup: logistic calibration fitted on the model's own training predictions and rho capped at 0.98. Old setup: raw probabilities, no rho cap. All three model seeds are shown; seed 42 remains the primary illustration.

Counts are annualized late-season equivalents under stationary resampling, not observed annual counts. Interval inclusion is exploratory; the binary-series correlation is used as a latent Gaussian AR coefficient.

| seed | method | sensor_rows | calibration_rows | reaches | covered | mean_width |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | platt_rho98 | 908 | 579 | 22 | 18 | 17.045 |
| 42 | raw_unclipped | 908 | 0 | 22 | 2 | 25.140 |
| 43 | platt_rho98 | 908 | 579 | 22 | 21 | 24.136 |
| 43 | raw_unclipped | 908 | 0 | 22 | 3 | 29.364 |
| 44 | platt_rho98 | 908 | 579 | 22 | 17 | 16.091 |
| 44 | raw_unclipped | 908 | 0 | 22 | 4 | 34.318 |
