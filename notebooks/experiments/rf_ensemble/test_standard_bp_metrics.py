import unittest
import numpy as np
import pandas as pd
from standard_bp_metrics import evaluate_predictions

class StandardMetricsTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame(dict(model=['m']*4,window_key=list('abcd'),record_key=['r1','r1','r1','r2'],
            overlap_group=['g1','g1','g1','g2'],sbp=[100.]*4,dbp=[60.]*4,
            prediction_sbp=[100.,105.,110.,115.],prediction_dbp=[60.,65.,70.,75.]))

    def test_inclusive_thresholds_and_metrics(self):
        r=evaluate_predictions(self.sample(),{'sbp':110.,'dbp':70.}).set_index('target')
        self.assertEqual(r.loc['sbp','within_5_mmHg_pct'],50)
        self.assertEqual(r.loc['sbp','within_10_mmHg_pct'],75)
        self.assertEqual(r.loc['sbp','within_15_mmHg_pct'],100)
        self.assertEqual(r.loc['sbp','mae_mmHg'],7.5)
        self.assertAlmostEqual(r.loc['sbp','rmse_mmHg'],np.sqrt(87.5))
        self.assertEqual(r.loc['sbp','group_macro_mae'],10)
        self.assertEqual(r.loc['sbp','mae_fraction_of_mean_baseline'],.75)

    def test_rejects_missing_and_duplicate_scores(self):
        f=self.sample();f.loc[0,'prediction_sbp']=np.nan
        with self.assertRaises(ValueError):evaluate_predictions(f,{'sbp':110,'dbp':70})
        f=self.sample();f.loc[1,'window_key']='a'
        with self.assertRaises(ValueError):evaluate_predictions(f,{'sbp':110,'dbp':70})

if __name__=='__main__':unittest.main()
