import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from expanded_data_analysis import choose_windows, grouped_split, grouped_correlations, quality_reasons
from random_forest_pipeline import load_data


class ExpandedAnalysisTests(unittest.TestCase):
    def test_random_record_sampling_is_reproducible_across_parts(self):
        first, _ = choose_windows(Path('.'),3,4,123)
        second, _ = choose_windows(Path('.'),3,4,123)
        pd.testing.assert_frame_equal(first,second)
        self.assertEqual(first.groupby('part').record_key.nunique().tolist(),[3]*4)
        self.assertFalse(first.duplicated(['part','record_id','window_id']).any())
        self.assertTrue((first.groupby('record_key').size()<=4).all())

    def test_shared_group_stays_together_even_across_parts(self):
        rows=[]
        for part in range(4):
            for record in range(30):
                rows.append(dict(part=f'Part_{part+1}',record_key=f'{part}:{record}',
                    overlap_group='shared' if record==0 else f'g{part}:{record}',
                    raw_ppg_sha256=f'h{part}:{record}',eligible=True))
        table,_=grouped_split(pd.DataFrame(rows),123)
        self.assertEqual(table[table.overlap_group.eq('shared')].split.nunique(),1)
        self.assertEqual(set(table.split),{'train','validation','test'})

    def test_grouped_correlations_reveal_opposite_within_record_pattern(self):
        rows=[]
        for r in range(6):
            for t in [-2,-1,0,1,2]:
                rows.append(dict(record_key=str(r),split='train',ppg_demo=10*r+t,
                                 sbp=100*r-t,dbp=50*r-t))
        # Reserved data must not enter the descriptive feature-ranking table.
        rows.append(dict(record_key='test',split='test',ppg_demo=-999,sbp=99999,dbp=99999))
        result=grouped_correlations(pd.DataFrame(rows),['ppg_demo'])
        self.assertTrue((result.pooled_spearman>0).all())
        self.assertTrue((result.record_median_spearman>0).all())
        self.assertTrue((result.within_record_spearman<-.99).all())
        self.assertEqual(result.complete_window_pairs.tolist(),[30,30])

    def test_quality_and_minimum_counts_are_real_gates(self):
        row=dict(processing_status='ok',ppg_quality_mean=.9,ppg_peak_count=5,
            abp_complete_beats=4,beat_count_mismatch=False,rate_mismatch=False,
            abp_detector_disagreement=False,label_reason='',sbp=120,dbp=70,
            nonfinite_samples=False,constant_signal=False,max_flat_extreme_samples=1,filter_boundary=False)
        self.assertEqual(quality_reasons(row),[])
        for change,expected in [({'ppg_quality_mean':.79},'ppg_quality_below_threshold'),
             ({'ppg_peak_count':2},'too_few_ppg_peaks'),
             ({'abp_complete_beats':2},'too_few_complete_abp_beats'),
             ({'beat_count_mismatch':True},'ppg_abp_peak_count_mismatch')]:
            self.assertIn(expected,quality_reasons(dict(row,**change)))

    def test_generated_data_has_no_known_cross_split_identity(self):
        table,features=load_data(Path('outputs/expanded_eda_oct14'))
        self.assertEqual(set(table.part),{f'Part_{i}' for i in range(1,5)})
        self.assertEqual(len(features),37)
        self.assertFalse(any('abp' in f or f in ['sbp','dbp','ppg_quality_mean'] for f in features))


if __name__=='__main__':
    unittest.main()
