import unittest
import numpy as np
import pandas as pd
from hja.copula import observed_period_counts


class ObservedCopulaTests(unittest.TestCase):
    def test_only_observed_dates_count(self):
        c=observed_period_counts([1,0,1],['2020-09-11','2020-09-20','2020-10-01'],.98,1)
        np.testing.assert_array_equal(c,np.full(10000,2))

    def test_probabilities_remain_date_specific(self):
        # Perfect latent correlation and ordered thresholds: E[count]=sum(p).
        c=observed_period_counts([.1,.4,.9],pd.date_range('2020-09-11',periods=3),1,1,n_sims=100000)
        self.assertAlmostEqual(c.mean(),1.4,delta=.015)
        self.assertAlmostEqual((c==3).mean(),.1,delta=.01)

    def test_calendar_gap_changes_dependence(self):
        rho=.8
        # For two .5-thresholded normal draws, count variance is
        # .5 + asin(latent correlation)/pi, and lag-three correlation=rho**3.
        c=observed_period_counts([.5,.5],['2020-09-11','2020-09-14'],rho,2,n_sims=100000)
        self.assertAlmostEqual(c.var(),.5+np.arcsin(rho**3)/np.pi,delta=.01)

    def test_reordering_preserves_pairs_and_rng(self):
        a=observed_period_counts([.1,.7],['2020-09-11','2020-09-16'],.7,3)
        b=observed_period_counts([.7,.1],['2020-09-16','2020-09-11'],.7,3)
        np.testing.assert_array_equal(a,b)

    def test_duplicate_dates_rejected(self):
        with self.assertRaises(ValueError):
            observed_period_counts([.1,.7],['2020-09-11','2020-09-11'],.7,3)


if __name__=='__main__':unittest.main()
