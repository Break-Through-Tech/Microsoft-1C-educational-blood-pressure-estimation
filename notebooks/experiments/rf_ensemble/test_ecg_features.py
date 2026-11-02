import unittest
import numpy as np
from prepare_ecg_comparison import ecg_features

class ECGFeatureTests(unittest.TestCase):
    def setUp(self):
        self.x=np.sin(2*np.pi*np.arange(1000)/125)
        self.r=np.arange(50,1000,125)
        self.pp=self.r+25

    def test_known_rate_and_cross_signal_delay(self):
        f,n,k=ecg_features(self.x,self.x,self.r,self.pp,125)
        self.assertAlmostEqual(f['ecg_rate_bpm'],60)
        self.assertAlmostEqual(f['ecg_rr_std_s'],0)
        self.assertAlmostEqual(f['ecg_ppg_peak_delay_median_s'],.2)
        self.assertAlmostEqual(f['ecg_ppg_peak_delay_iqr_s'],0)
        self.assertEqual(f['ecg_ppg_peak_match_fraction'],1)
        self.assertGreater(n,0);self.assertGreater(k,0)

    def test_normalized_shape_is_gain_offset_invariant(self):
        a,_,_=ecg_features(self.x,self.x,self.r,self.pp,125)
        b,_,_=ecg_features(4*self.x+9,4*self.x+9,self.r,self.pp,125)
        for name in a:
            if name.startswith('ecg_z_'):self.assertAlmostEqual(a[name],b[name],places=9)

    def test_no_r_peaks_leaves_timing_missing(self):
        f,n,k=ecg_features(self.x,self.x,np.array([],dtype=int),self.pp,125)
        self.assertEqual((n,k),(0,0))
        self.assertTrue(np.isnan(f['ecg_rate_bpm']))
        self.assertTrue(np.isnan(f['ecg_ppg_peak_delay_median_s']))
        self.assertEqual(f['ecg_ppg_peak_match_fraction'],0)

if __name__=='__main__':unittest.main()
