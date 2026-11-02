"""Common blood-pressure regression report for saved predictions.

Percent-within thresholds is error coverage, not classification accuracy.
Thresholds include their boundary: |prediction - reference| <= threshold.
Baseline constants must be fitted on TRAINING data, never on evaluated labels.
Pass one split/modality at a time, or use distinct model names for each setting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

TARGETS=['sbp','dbp']


def evaluate_predictions(frame,training_means):
    required=['model','window_key','record_key','overlap_group']+TARGETS+['prediction_'+t for t in TARGETS]
    if not set(required)<=set(frame):raise ValueError('Missing prediction schema columns')
    if frame[required].isna().any().any():raise ValueError('Missing IDs, labels or predictions; report exclusions explicitly before scoring')
    rows=[]
    for model,data in frame.groupby('model',sort=False):
        if data.window_key.duplicated().any():raise ValueError('Duplicate window keys within a model')
        for t in TARGETS:
            truth=data[t].to_numpy(dtype=float);guess=data['prediction_'+t].to_numpy(dtype=float)
            if not np.isfinite(truth).all() or not np.isfinite(guess).all():raise ValueError('Nonfinite labels or predictions')
            e=guess-truth;ae=np.abs(e);baseline=np.abs(float(training_means[t])-truth).mean()
            meta=data[['record_key','overlap_group']].copy();meta['ae']=ae
            row=dict(model=model,target=t,windows=len(data),recordings=data.record_key.nunique(),overlap_groups=data.overlap_group.nunique(),
                mae_mmHg=float(ae.mean()),rmse_mmHg=float(np.sqrt(np.mean(e**2))),r2=float(r2_score(truth,guess)) if len(truth)>1 else np.nan,
                bias_mmHg=float(e.mean()),error_sd_mmHg=float(e.std(ddof=1)) if len(e)>1 else np.nan,
                recording_macro_mae=float(meta.groupby('record_key').ae.mean().mean()),
                group_macro_mae=float(meta.groupby('overlap_group').ae.mean().mean()),
                mean_baseline_mae=float(baseline),mae_fraction_of_mean_baseline=float(ae.mean()/baseline) if baseline>0 else np.nan)
            for threshold in [5,10,15]:row[f'within_{threshold}_mmHg_pct']=float(100*np.mean(ae<=threshold))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions');p.add_argument('--training-means-json',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();means=json.loads(open(a.training_means_json).read())
    result=evaluate_predictions(pd.read_csv(a.predictions),means);result.to_csv(a.output,index=False)
    print(result.round(3).to_string(index=False))

if __name__=='__main__':main()
