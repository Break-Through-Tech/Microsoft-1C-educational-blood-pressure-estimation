"""Checks for the new pulse measurements, independent of the raw dataset."""
import unittest
import numpy as np
from preprocessing_ablation import pulse_features,zscore

class PulseFeatureTests(unittest.TestCase):
    def setUp(self):
        self.x=-np.cos(2*np.pi*np.arange(625)/125)
        self.peaks=np.array([62,187,312,437,562])

    def test_shape_features_ignore_positive_gain_and_offset(self):
        first,n=pulse_features(self.x,self.peaks)
        second,m=pulse_features(7*self.x+42,self.peaks)
        self.assertEqual(n,m)
        self.assertGreaterEqual(n,2)
        for name in first:
            self.assertAlmostEqual(first[name],second[name],places=8)

    def test_pulse_timing_and_nested_widths(self):
        f,n=pulse_features(self.x,self.peaks)
        self.assertEqual(n,3)
        self.assertAlmostEqual(f['pulse_duration_s_median'],1)
        self.assertAlmostEqual(f['pulse_rate_bpm'],60)
        self.assertAlmostEqual(f['pulse_rise_s_median']+f['pulse_fall_s_median'],1)
        self.assertGreater(f['pulse_width_25_s_median'],f['pulse_width_50_s_median'])
        self.assertGreater(f['pulse_width_50_s_median'],f['pulse_width_75_s_median'])

    def test_incomplete_or_constant_signals_do_not_invent_shape(self):
        for signal,peaks in [(self.x,self.peaks[:2]),(np.ones(625),self.peaks)]:
            f,n=pulse_features(signal,peaks)
            self.assertEqual(n,0)
            self.assertTrue(np.isnan(f['pulse_height_median']))
        self.assertTrue(np.isnan(zscore(np.ones(625))).all())

if __name__=='__main__':unittest.main()
