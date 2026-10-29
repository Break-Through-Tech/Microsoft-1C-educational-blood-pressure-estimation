"""Input checks for the PPG-only prediction entry point."""
import unittest
import numpy as np
from predict_ppg_rf import window_features

class PredictionInputTests(unittest.TestCase):
    def test_rejects_invalid_recordings_before_processing(self):
        for signal in [np.ones(1000),np.full(1000,np.nan),np.zeros((1000,2))]:
            with self.assertRaises(ValueError):window_features(signal,125)

    def test_requires_training_filter_context(self):
        signal=np.sin(np.arange(1000)/10)
        for start in [0,124,251,125.5]:
            with self.assertRaises(ValueError):window_features(signal,start)

if __name__=='__main__':unittest.main()
