import unittest
import numpy as np
from predict_ppg_ecg import predict

class CombinedPredictionInputTests(unittest.TestCase):
    def test_requires_synchronized_equal_length_arrays(self):
        with self.assertRaises(ValueError):predict(np.arange(1000),np.arange(900),125,'unused.joblib')
        with self.assertRaises(ValueError):predict(np.zeros((1000,2)),np.arange(1000),125,'unused.joblib')

    def test_rejects_bad_ecg_before_loading_model(self):
        for ecg in [np.ones(1000),np.full(1000,np.nan)]:
            with self.assertRaises(ValueError):predict(np.arange(1000),ecg,125,'unused.joblib')

if __name__=='__main__':unittest.main()
