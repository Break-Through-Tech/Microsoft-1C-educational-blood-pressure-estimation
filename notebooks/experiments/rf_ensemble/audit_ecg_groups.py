"""Check exact ECG record/window repeats before the matched-channel comparison.

Retain validation windows. Exclude whole training groups connected to validation
by either existing PPG groups or newly detected exact nonconstant ECG repeats.
This does not establish patient identity or exhaust all shifted ECG overlaps.
"""
from pathlib import Path
import hashlib,json
import h5py
import numpy as np
import pandas as pd
from expanded_data_analysis import ConnectedRecords

OUT=Path('outputs/rf_ecg_features_new')


def digest(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()


def main():
    meta=pd.read_csv(OUT/'rows_and_labels.csv')
    graph=ConnectedRecords(meta.overlap_group.unique())
    seen_record={};seen_window={};matches=[];hashes=[]
    for (part,key),rows in meta.groupby(['part','record_key']):
        with h5py.File(Path('data')/(part+'.mat'),'r') as h:
            x=np.asarray(h[h[part][int(key.split(':')[-1]),0]][:,2],dtype=float)
        group=rows.overlap_group.iloc[0]
        if np.isfinite(x).all() and np.ptp(x)>0:
            d=digest(x)
            if d in seen_record:
                other,other_key=seen_record[d];graph.union(group,other)
                matches.append(dict(kind='record',record_key=key,other_record=other_key))
            else:seen_record[d]=(group,key)
        for i,s in rows.iterrows():
            start=int(s.window_key.split(':')[-1])*625;window=x[start:start+625]
            if not np.isfinite(window).all() or np.ptp(window)==0:continue
            d=digest(window);hashes.append(dict(window_key=s.window_key,ecg_sha256=d))
            if d in seen_window:
                other,other_key=seen_window[d];graph.union(group,other)
                matches.append(dict(kind='window',record_key=key,other_record=other_key))
            else:seen_window[d]=(group,key)
    meta['feature_row']=meta.index
    meta['ppg_overlap_group']=meta.overlap_group
    meta['overlap_group']=meta.overlap_group.map(graph.find)
    held=set(meta.loc[meta.split.eq('validation'),'overlap_group'])
    blocked=meta.split.eq('train')&meta.overlap_group.isin(held)
    meta['excluded_ecg_overlap']=blocked
    meta.to_csv(OUT/'cohort_audit.csv',index=False)
    meta[~blocked].to_csv(OUT/'matched_cohort.csv',index=False)
    pd.DataFrame(matches,columns=['kind','record_key','other_record']).to_csv(OUT/'ecg_exact_matches.csv',index=False)
    pd.DataFrame(hashes).to_csv(OUT/'ecg_window_hashes.csv',index=False)
    summary=dict(exact_match_events=len(matches),training_windows_excluded=int(blocked.sum()),
        validation_windows=int(meta.split.eq('validation').sum()),
        training_windows=int((meta.split.eq('train')&~blocked).sum()),
        groups_merged=int(meta.ppg_overlap_group.nunique()-meta.overlap_group.nunique()),
        limitation='Exact nonconstant ECG full records and selected windows only; shifted/near ECG overlaps not exhaustively checked')
    (OUT/'ecg_overlap_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=OUT)
    OUT=parser.parse_args().data_dir
    main()
