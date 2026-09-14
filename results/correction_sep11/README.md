# September 11 correction campaign

Launched on branch `correction-sep11`. Nine training runs, seeds 42/43/44:
corrected LSTM control, corrected LSTM with status availability, and RGCN
with status availability. Existing q65 RGCN checkpoints are reused as controls.
GPU 0; detached supervisor PID is recorded in `launch.json` and `status.json`.

`status.json` records the current stage, child PID, completed stages, and any
failure. `queue.log` tracks progress; each stage has its own log. The queue
stops on a failed stage instead of silently continuing. It automatically
exports RGCN predictions and computes matched metrics when all runs finish.

The LSTM uses 30 consecutive calendar days and exact calendar t+3 targets.
Depth and weather filling is forward-only within reach; remaining feature
missingness is zero-filled after train-only normalization. Unknown status is
not forward-filled. The target label is sensor status where available,
otherwise dry for discharge <=0.00014 CMS, otherwise missing and excluded
from loss. The availability flag distinguishes known status (including
discharge-imputed dry) from missing status. The RGCN flag is lagged and
forecast-tail-frozen with its status input. Existing RGCN feature arrays are
read-only inputs; the additional channel is derived at load time from their
unfilled wet/dry targets. No base arrays or canonical checkpoints are overwritten.

LSTM hyperparameters and internal training-side early stopping split are
retained; best weights now use cloned tensors. No resampling, threshold
0.5, q65 cutoff 2020-09-10. Calendar reconstruction changes the training pool:
8,454 labeled sequences (6,836 dry / 1,618 wet); 1,072 validation sequences,
including 956 sensor targets. Details are in `preflight_data.json`.

Five focused CPU checks passed, plus a batched RGCN forward/backward smoke
check with the added channel and forecast masking. Source SHA-256 hashes
at launch are recorded in `launch.json`. Data is under
`data/retrain/correction_sep11`; per-variant LSTM predictions are here.

Final classification scoring joins exact raw sensor reach/date keys and
uses common issue/target dates across every model, variant, and seed.
Mean and sample SD across seeds describe training variability, not sampling
uncertainty. Discharge diagnostics use the existing stride-3 horizon convention.

The no-retraining copula correction is already complete:
`copula_sensor_only.json` and `copula_sensor_only.csv`. Exact sensor-key
filtering gives 908 validation rows, 579 training calibration rows, and
19/22 interval inclusion. This remains an annualized late-season equivalent
with calibration fitted on the base model's training predictions; it is
not observed annual-count validation. Canonical manuscript files are unchanged.
