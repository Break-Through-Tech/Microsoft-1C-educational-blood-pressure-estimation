"""Summarize saved preprocessing predictions without fitting models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(output=Path('outputs/preprocessing_ablation_oct28')):
    output=Path(output)
    m=pd.read_csv(output/'metrics.csv');p=pd.read_csv(output/'predictions.csv')
    rows=[]
    for split in ['training_oof','validation']:
        for model in ['RF','Boosting']:
            sub=p[p.split.eq(split)&p.model.eq(model)]
            for target in ['sbp','dbp']:
                def errors(frame):
                    return frame.assign(ae=(frame['prediction_'+target]-frame[target]).abs()).groupby('overlap_group').ae.mean()
                base=errors(sub[sub.variant.eq('Original 37')])
                for variant in sub.variant.unique():
                    new=errors(sub[sub.variant.eq(variant)]).reindex(base.index)
                    assert new.notna().all()
                    d=(base-new).to_numpy();rng=np.random.default_rng(20261028)
                    boot=d[rng.integers(len(d),size=(2000,len(d)))].mean(axis=1)
                    rows.append(dict(split=split,model=model,variant=variant,target=target,
                        gain_mmHg=d.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
    pd.DataFrame(rows).to_csv(output/'paired_group_gains.csv',index=False)
    rank=m.query('split=="training_oof" and variant!="Constant"').groupby(['variant','model']).overlap_group_macro_mae.mean().sort_values()
    (output/'training_ranking.json').write_text(json.dumps([dict(variant=k[0],model=k[1],mean_target_group_mae=float(v)) for k,v in rank.items()],indent=2))
    order=['Original 37','Normalized 37','NK pulse shape','Original + NK pulse shape','Bandpass pulse shape']
    labels=['Original','Normalized original','Complete-pulse shape','Original + pulse shape','Bandpass pulse shape']
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,model in enumerate(['RF','Boosting']):
        for j,target in enumerate(['sbp','dbp']):
            ax=axes[i,j];v=m.query('split=="validation" and model==@model and target==@target').set_index('variant').reindex(order)
            ax.barh(labels,v.mae_mmHg,color=['#6b7280','#418d95','#a899bb','#5f9471','#c29b62']);ax.invert_yaxis()
            for k,n in enumerate(v.mae_mmHg):ax.text(n+.12,k,f'{n:.2f}',va='center')
            ax.set_xlim(0,v.mae_mmHg.max()+2);ax.set_title(model+' — '+target.upper());ax.set_xlabel('Validation MAE (mmHg)')
    fig.suptitle('Same 857 validation windows, pressure labels and fixed model settings')
    fig.tight_layout();fig.savefig(output/'validation_comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':summarize()
