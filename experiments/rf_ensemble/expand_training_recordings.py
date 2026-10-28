"""Expand training recordings with unchanged gates and a fixed validation set.

New recordings are sampled without targets. Original selected records and their
known overlap groups are ineligible for sampling. A full exact/shifted-overlap
rescan precedes training. Test data is used only by the signal-overlap audit,
never for targets, feature development, model selection or error evaluation.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from expanded_data_analysis import process_windows,group_overlaps

OUT=Path('outputs/training_expansion_oct28')
SOURCE=Path('outputs/expanded_eda_oct14')
SEED=20261028


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    old=pd.read_csv(SOURCE/'expanded_features_and_split.csv')
    inventory=pd.read_csv(SOURCE/'all_parts_inventory_and_groups.csv')
    protected_groups=set(inventory.loc[inventory.record_key.isin(old.record_key),'overlap_group'])
    candidates=inventory[~inventory.overlap_group.isin(protected_groups)].copy()
    rng=np.random.default_rng(SEED)
    new_records=[];rows=[]
    for part,part_rows in candidates.groupby('part'):
        chosen=part_rows.iloc[rng.choice(len(part_rows),250,replace=False)].sort_values('record_id')
        for s in chosen.itertuples():
            new_records.append(dict(part=part,record_id=s.record_id,record_key=s.record_key))
            windows=np.sort(rng.choice(int(s.complete_windows),min(12,int(s.complete_windows)),replace=False))
            for wid in windows:
                rows.append(dict(part=part,record_id=int(s.record_id),record_key=s.record_key,
                    window_id=int(wid),start_sample=int(wid*625),analysis_pool='random_all_parts'))
    selection=pd.DataFrame(rows)
    selection.to_csv(OUT/'new_window_selection.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps(dict(seed=SEED,new_recordings_per_part=250,
        maximum_windows_per_record=12,selection='Uniform source recording sample per Part, no targets',
        gates='Unchanged October 14 gates and labels',validation='Fixed original validation windows',
        test='No test feature extraction or evaluation; source PPG only for overlap detection',
        planned_models='Fixed original RF and fixed boosting; original, normalized and original-plus-pulse features',
        learning_curve='Original training groups, plus first half, plus all eligible new groups in seeded nested order',
        caution='Exploratory validation; repeated windows are not independent patients'),indent=2))
    table,features=process_windows(Path.cwd(),selection,inventory,OUT)
    # Include original anchors as well, preserving previous overlap protections.
    anchor_records=pd.concat([old[['part','record_id','record_key']],pd.DataFrame(new_records)],ignore_index=True).drop_duplicates()
    graph=group_overlaps(Path.cwd(),anchor_records,inventory.copy(),OUT)
    held_records=set(old.loc[old.split.isin(['validation','test']),'record_key'])
    held_groups={graph.find(r) for r in held_records}
    new=table[table.analysis_pool.eq('random_all_parts')].copy()
    new['overlap_group']=new.record_key.map(graph.find)
    new['split']='train'
    new['blocked_overlap_with_heldout']=new.overlap_group.isin(held_groups)
    new.loc[new.blocked_overlap_with_heldout,'eligible']=False
    original_train=old[old.eligible & old.split.eq('train')].copy()
    original_train['overlap_group']=original_train.record_key.map(graph.find)
    original_train['blocked_overlap_with_heldout']=original_train.overlap_group.isin(held_groups)
    removed=original_train[original_train.blocked_overlap_with_heldout]
    original_train=original_train[~original_train.blocked_overlap_with_heldout].copy()
    validation=old[old.eligible & old.split.eq('validation')].copy()
    validation['overlap_group']=validation.record_key.map(graph.find)
    # Also protect old test identities from the fixed validation evaluation.
    test_groups={graph.find(r) for r in old.loc[old.split.eq('test'),'record_key']}
    if validation.overlap_group.isin(test_groups).any():
        raise ValueError('New overlap evidence connects fixed validation and old test; review before modeling')
    combined=pd.concat([original_train,new[new.eligible],validation],ignore_index=True)
    duplicate=combined.duplicated('raw_ppg_sha256',keep='first')
    if (duplicate & combined.split.eq('validation')).any():
        raise ValueError('Exact training/validation duplicate after overlap audit')
    combined=combined[~duplicate].copy()
    combined['eligible']=True
    for identity in ['record_key','overlap_group','raw_ppg_sha256']:
        assert combined.groupby(identity).split.nunique().max()==1
    combined['training_origin']=np.where(combined.record_key.isin(original_train.record_key),'original','new')
    new.to_csv(OUT/'new_window_audit.csv',index=False)
    removed.to_csv(OUT/'original_training_overlap_exclusions.csv',index=False)
    combined.to_csv(OUT/'expanded_features_and_split.csv',index=False)
    (OUT/'feature_columns.json').write_text(json.dumps(features,indent=2))
    summary=dict(new_sampled_records=len(new_records),new_sampled_windows=len(new),
        new_quality_eligible_before_overlap=int(table.loc[table.analysis_pool.eq('random_all_parts'),'eligible'].sum()),
        new_training_windows_after_all_exclusions=int(combined.split.eq('train').sum())-len(original_train),
        new_quality_eligible_windows=int(new.eligible.sum()),new_heldout_overlap_windows=int(new.blocked_overlap_with_heldout.sum()),
        original_training_windows_excluded=len(removed),original_training_windows=len(original_train),
        total_training_windows=int(combined.split.eq('train').sum()),
        total_training_records=int(combined[combined.split.eq('train')].record_key.nunique()),
        validation_windows=len(validation),test_evaluated=False)
    (OUT/'preparation_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    OUT=parser.parse_args().output_dir
    main()
