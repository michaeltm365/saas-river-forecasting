import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from correction_lstm import sequences
from rgcn.pipeline import availability, features as F
from rgcn.pipeline.masking import mask_forecast_tail


class CorrectionChecks(unittest.TestCase):
    def test_unknown_is_distinct_from_known_dry(self):
        wet = torch.tensor([[float("nan")], [0.], [1.], [float("nan")]])
        x = availability.append_channel(torch.zeros(4, 1, 20), wet)
        self.assertEqual(x[:, 0, 20].tolist(), [0., 0., 1., 1.])

    def test_future_status_cannot_change_tail(self):
        wet = torch.arange(36).float().remainder(2).reshape(-1, 1)
        wet[29] = float("nan")
        a = availability.append_channel(torch.randn(36, 1, 20), wet)
        b = a.clone()
        # At issue date index 29, obs at 30 and 31 are unknown future.
        b[31:, :, 20] = 1 - b[31:, :, 20]
        a = mask_forecast_tail(a[:33].clone(), 30, "obs+drivers", 20)
        b = mask_forecast_tail(b[:33].clone(), 30, "obs+drivers", 20)
        torch.testing.assert_close(a, b)
        self.assertEqual(a[30:, 0, 20].tolist(), [0., 0., 0.])

    def test_existing_features_identical_with_indicator(self):
        base = torch.randn(2, 33, 3, 20)
        aug = torch.cat([base, torch.ones(2, 33, 3, 1)], dim=-1)
        old = mask_forecast_tail(base.clone(), 30, "obs+drivers")
        new = mask_forecast_tail(aug, 30, "obs+drivers", 20)
        torch.testing.assert_close(old, new[..., :20])
        keep, names = availability.selection(F.DISCHARGE_LAG_VARS[1:] + F.WETDRY_LAG_VARS[1:], True)
        self.assertEqual(len(names), 36)
        self.assertEqual(keep[-1], 20)
        self.assertEqual(names[len(keep) - 1], availability.NAME)

    def test_calendar_target_and_last_window(self):
        dates = pd.date_range("2020-08-01", periods=35)
        df = pd.DataFrame(dict(NHDPlusID=1, Date=dates, target_date=dates + pd.Timedelta(days=3),
                               wet_dry_next=[np.nan] * 29 + [0, 1, 0, np.nan, np.nan, np.nan],
                               label_is_hobo=1, feature=np.arange(35)))
        x, y, sites, td, hobo = sequences(df, ["feature"])
        self.assertEqual(x.shape, (3, 30, 1))
        self.assertEqual(td[-1], dates[34])
        self.assertEqual(x[-1, -1, 0], 31)
        self.assertTrue(np.isfinite(y).all())

    def test_cloned_checkpoint_is_immutable(self):
        model = torch.nn.Linear(2, 1)
        best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        original = best["weight"].clone()
        with torch.no_grad():
            model.weight.add_(1)
        torch.testing.assert_close(best["weight"], original)
        model.load_state_dict(best)
        torch.testing.assert_close(model.weight, original)


if __name__ == "__main__":
    unittest.main()
