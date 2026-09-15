import unittest
import numpy as np
import pandas as pd
from hja.data import feature_frame
from hja.models.lstm_hobo import temporal_masks


class HoboTemporalTests(unittest.TestCase):
    def test_causal_features_never_use_later_rows_or_other_sites(self):
        df = pd.DataFrame({'NHDPlusID': [1, 1, 1, 2],
                           'Date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-01']),
                           'depth': [np.nan, 2., np.nan, np.nan]})
        x, _ = feature_frame(df, causal=True)
        np.testing.assert_array_equal(x.depth, [0., 2., 2., 0.])
        df.loc[2, 'depth'] = 99.
        altered, _ = feature_frame(df, causal=True)
        np.testing.assert_array_equal(x.depth[:2], altered.depth[:2])

    def test_target_and_issue_boundaries(self):
        targets = pd.date_range('2020-08-01', periods=65)
        issues = targets - pd.Timedelta(days=3)
        fit, val, test, inner = temporal_masks(targets, '2020-09-15', issues)
        self.assertTrue((targets[fit] < inner).all())
        self.assertTrue((issues[val] >= inner).all())
        self.assertTrue((targets[val] < pd.Timestamp('2020-09-15')).all())
        self.assertTrue((issues[test] >= pd.Timestamp('2020-09-15')).all())
        self.assertFalse((fit & val | fit & test | val & test).any())
        self.assertLess(targets[fit].max(), issues[val].min())
        self.assertLess(targets[val].max(), issues[test].min())

    def test_same_day_targets_excluded(self):
        targets = pd.date_range('2020-08-01', periods=65)
        issues = targets - pd.Timedelta(days=1)
        issues = issues.to_numpy().copy()
        issues[0] = targets[0].to_datetime64()
        fit, _, _, _ = temporal_masks(targets, '2020-09-15', issues)
        self.assertFalse(fit[0])

class CalendarTargetTests(unittest.TestCase):
    def test_exact_calendar_lookup_preserves_history_and_missing_targets(self):
        from hja.data import attach_hobo_calendar_targets
        obs = pd.DataFrame({'NHDPlusID': [1,1,1,1,2],
                            'Date': pd.to_datetime(['2020-01-01','2020-01-03','2020-01-04','2020-01-08','2020-01-04']),
                            'HoboWetDry0.05': [0.,1.,1.,0.,0.]})
        # The target date can exist in raw observations but be absent after joins.
        frame = obs.iloc[[0,1,3]][['NHDPlusID','Date']]
        got = attach_hobo_calendar_targets(frame, obs)
        self.assertEqual(len(got),3)
        self.assertEqual(got.wet_dry_next.iloc[0],1.)
        self.assertTrue(got.wet_dry_next.iloc[1:].isna().all())
        self.assertTrue((got.target_date-got.Date).eq(pd.Timedelta(days=3)).all())

    def test_ambiguous_observed_labels_rejected(self):
        from hja.data import attach_hobo_calendar_targets
        obs = pd.DataFrame({'NHDPlusID':[1,1], 'Date':pd.to_datetime(['2020-01-04']*2),
                            'HoboWetDry0.05':[0.,1.]})
        with self.assertRaisesRegex(ValueError,'unique reach/date'):
            attach_hobo_calendar_targets(obs[['NHDPlusID','Date']],obs)


if __name__ == '__main__':
    unittest.main()
