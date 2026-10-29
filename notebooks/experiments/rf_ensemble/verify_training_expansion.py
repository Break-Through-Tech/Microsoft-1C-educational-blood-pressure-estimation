"""Verify sampling and fixed-validation invariants for the expansion run."""
from pathlib import Path
import numpy as np
import pandas as pd
from random_forest_pipeline import load_data,TARGETS


def verify():
    source=Path('outputs/expanded_eda_oct14')
    data=Path('outputs/training_expansion_oct28')
    old=pd.read_csv(source/'expanded_features_and_split.csv')
    table,features=load_data(data)
    assert not table.split.eq('test').any()
    expected=old[old.eligible & old.split.eq('validation')].set_index('window_key').sort_index()
    actual=table[table.split.eq('validation')].set_index('window_key').sort_index()
    assert expected.index.equals(actual.index)
    for name in TARGETS+features:
        np.testing.assert_allclose(expected[name],actual[name],rtol=1e-12,atol=1e-12,equal_nan=True)
    assert expected.raw_ppg_sha256.equals(actual.raw_ppg_sha256)
    sample=pd.read_csv(data/'new_window_selection.csv')
    selected=sample[['part','record_key']].drop_duplicates()
    assert selected.groupby('part').size().tolist()==[250]*4
    assert not set(sample.record_key)&set(old.record_key)
    assert sample.groupby('record_key').size().max()<=12
    assert not sample.duplicated('window_key').any() if 'window_key' in sample else not sample.duplicated(['record_key','window_id']).any()
    assert table.groupby('record_key').size().max()<=12
    inventory=pd.read_csv(data/'all_parts_inventory_and_groups.csv').set_index('record_key')
    held=set(inventory.loc[old.loc[old.split.isin(['validation','test']),'record_key'].unique(),'overlap_group'])
    assert not set(table.loc[table.split.eq('train'),'overlap_group'])&held
    curve=Path('outputs/training_learning_curve_oct28')
    if (curve/'nested_groups.csv').exists():
        groups=pd.read_csv(curve/'nested_groups.csv')
        sets={k:set(v.overlap_group) for k,v in groups.groupby('level')}
        assert sets['Original training']<=sets['Half expansion']<=sets['Full expansion']
    pred_path=curve/'predictions.csv'
    if not pred_path.exists():pred_path=curve/'predictions.csv.gz'
    if pred_path.exists():
        pred=pd.read_csv(pred_path)
        for (_,_,_),rows in pred[pred.split.eq('validation')].groupby(['level','variant','model']):
            got=rows.set_index('window_key').sort_index()
            assert got.index.equals(expected.index)
            np.testing.assert_allclose(got[TARGETS],expected[TARGETS],rtol=1e-12,atol=1e-12)
    print('Sampling, held-out overlap exclusions, fixed validation, and available learning-curve invariants passed.')

if __name__=='__main__':verify()
