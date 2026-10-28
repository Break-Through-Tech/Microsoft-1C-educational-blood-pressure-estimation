"""Shareable plots for the all-Part EDA and saved RF validation results."""
from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_resnet_data import waveform,paired_labels,FS,WIDTH

ROOT=Path(__file__).resolve().parent
EDA=ROOT/'outputs/expanded_eda_oct14'
RF=ROOT/'outputs/random_forest_oct14'
COLORS=['#217b84','#e4953b','#785ba6','#4979b4']


def save(fig,name,folder=EDA):
    fig.tight_layout()
    fig.savefig(folder/name,dpi=150,bbox_inches='tight')
    plt.close(fig)


def eda_plots():
    table=pd.read_csv(EDA/'expanded_features_and_split.csv')
    audit=pd.read_csv(EDA/'all_sampled_window_audit.csv')
    parts=pd.read_csv(EDA/'part_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(parts.part,parts.eligible_windows,color='teal',label='Provisionally retained')
    axes[0].bar(parts.part,parts.selected_windows-parts.eligible_windows,bottom=parts.eligible_windows,color='lightgray',label='Needs review / duplicate')
    axes[0].set(title='Random recording sample across all four Parts',ylabel='Five-second windows')
    axes[0].legend(fontsize=8)
    for i,(part,group) in enumerate(table[table.eligible].groupby('part')):
        counts=group.split.value_counts().reindex(['train','validation','test'],fill_value=0)
        axes[1].bar(np.arange(3)+(i-1.5)*.18,counts,width=.18,color=COLORS[i],label=part)
    axes[1].set(xticks=np.arange(3),xticklabels=['Train','Validation','Reserved test'],ylabel='Retained windows',title='Overlap-group splits; each Part represented')
    axes[1].legend(fontsize=8)
    save(fig,'sampling_and_splits.png')

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    legacy=audit[audit.analysis_pool.eq('legacy_first500')]
    random=audit[audit.analysis_pool.eq('random_all_parts')]
    for data,label,color in [(legacy,'First 500 (six recordings)','darkorange'),(random,'Random sample (800 recordings)','teal')]:
        axes[0].hist(data.ppg_rate_bpm.dropna(),bins=np.arange(20,211,5),density=True,alpha=.5,label=label,color=color)
    axes[0].set(title='Same full-record PPG detector for both samples',xlabel='PPG pulse rate (BPM)',ylabel='Density')
    axes[0].legend(fontsize=8)
    grouped=[table.loc[table.part.eq(p)&table.eligible,'ppg_rate_bpm'] for p in sorted(table.part.unique())]
    axes[1].boxplot(grouped,tick_labels=sorted(table.part.unique()),showfliers=True)
    axes[1].set(title='Retained pulse-rate variation by Part',ylabel='PPG pulse rate (BPM)')
    save(fig,'pulse_rate_sampling_comparison.png')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for eligible,label,color in [(False,'Withheld','lightgray'),(True,'Provisionally retained','teal')]:
        rows=table[table.eligible.eq(eligible)]
        axes[0].scatter(rows.abp_peak_count,rows.ppg_peak_count,color=color,s=16,alpha=.25,label=label)
        axes[1].scatter(rows.abp_rate_bpm,rows.ppg_rate_bpm,color=color,s=12,alpha=.2,label=label)
    axes[0].plot([0,18],[0,18],'k:',label='Equal counts')
    axes[0].set(xlabel='ABP peaks in the same five seconds',ylabel='PPG peaks in the same five seconds',title='Peak-count comparison')
    limit=210
    axes[1].plot([0,limit],[0,limit],'k:',label='Equal rates')
    axes[1].plot([0,limit/2],[0,limit],'--',color='darkorange',label='PPG rate = 2 x ABP')
    axes[1].plot([0,limit],[0,limit/2],'--',color='purple',label='ABP rate = 2 x PPG')
    axes[1].set(xlim=(20,limit),ylim=(20,limit),xlabel='ABP interval-derived rate (BPM)',ylabel='PPG interval-derived rate (BPM)',title='Doubled-rate flags require waveform review')
    for ax in axes:ax.legend(fontsize=7)
    save(fig,'ppg_abp_beat_comparison.png')

    sensitivity=pd.read_csv(EDA/'quality_threshold_sensitivity.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,(part,group) in enumerate(sensitivity.groupby('part')):
        axes[0].plot(group.quality_threshold,group.retained_windows,'o-',color=COLORS[i],label=part)
    axes[0].axvline(.8,color='gray',linestyle=':')
    axes[0].set(xlabel='Minimum mean PPG_Quality',ylabel='Retained windows (all other gates fixed)',title='Quality-threshold sensitivity',xticks=[.6,.7,.8,.9])
    axes[0].legend(fontsize=8)
    for part,group in table.groupby('part'):
        axes[1].hist(group.ppg_quality_mean,bins=np.linspace(0,1,31),histtype='step',label=part)
    axes[1].axvline(.8,color='gray',linestyle='--')
    axes[1].set(xlabel='Mean full-record template quality in each window',ylabel='Selected windows',title='Quality is template similarity, not correctness')
    axes[1].legend(fontsize=8)
    save(fig,'quality_threshold_sensitivity.png')

    corr=pd.read_csv(EDA/'development_grouped_correlations.csv')
    ordered=corr[corr.target.eq('sbp')].set_index('feature').record_median_spearman.abs().nlargest(12).index
    fig,axes=plt.subplots(1,3,figsize=(13,6))
    for ax,view,title in zip(axes,['pooled_spearman','record_median_spearman','within_record_spearman'],['Pooled windows','Recording medians','Within-record centered']):
        matrix=corr.pivot(index='feature',columns='target',values=view).reindex(index=ordered,columns=['sbp','dbp'])
        chart=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
        ax.set(xticks=[0,1],xticklabels=['SBP','DBP'],yticks=range(len(ordered)),yticklabels=ordered if ax==axes[0] else ['']*len(ordered),title=title)
        for r in range(len(ordered)):
            for c in range(2):ax.text(c,r,f'{matrix.iloc[r,c]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(chart,ax=axes.ravel().tolist(),label='Spearman correlation',fraction=.025,pad=.04)
    fig.suptitle('Development data only; selected by |record-median SBP correlation|',y=1.01)
    # Explicit spacing leaves room for feature labels and the shared colorbar.
    fig.subplots_adjust(left=.25,right=.88,wspace=.2,top=.90,bottom=.08)
    fig.savefig(EDA/'grouped_feature_correlations.png',dpi=150,bbox_inches='tight')
    plt.close(fig)

    examples=pd.concat([
        table[table.possible_ppg_double_count].head(2),
        table[table.possible_abp_double_count].head(2),
        table[table.ppg_quality_mean.lt(.8)&table.ppg_quality_mean.notna()].head(1),
        legacy[legacy.record_id.eq(0)&legacy.window_id.eq(0)],
        table[table.eligible&table.split.eq('train')].sample(2,random_state=14),
    ],ignore_index=True).drop_duplicates('window_key')
    examples.to_csv(EDA/'beat_review_examples.csv',index=False)
    for page in range((len(examples)+3)//4):
        sample=examples.iloc[page*4:(page+1)*4]
        fig,axes=plt.subplots(len(sample),2,figsize=(15,len(sample)*3.2),squeeze=False)
        for row,s in enumerate(sample.itertuples()):
            with h5py.File(ROOT/'data'/f'{s.part}.mat','r') as handle:ppg,abp,_=waveform(handle,s.record_id)
            sig,info=nk.ppg_process(ppg,sampling_rate=FS)
            pp=np.asarray(info['PPG_Peaks']);bp,_=paired_labels(abp,10)
            start,stop=int(s.start_sample),int(s.start_sample+WIDTH)
            pp=pp[(pp>=start)&(pp<stop)];bp=bp[(bp>=start)&(bp<stop)]
            t=np.arange(WIDTH)/FS
            axes[row,0].plot(t,sig.PPG_Clean.iloc[start:stop],color='teal')
            axes[row,0].scatter((pp-start)/FS,sig.PPG_Clean.iloc[pp],color='darkorange',s=20)
            axes[row,0].set(title=f'{s.window_key}: PPG {len(pp)} peaks; quality {s.ppg_quality_mean:.2f}',ylabel='Clean PPG')
            axes[row,1].plot(t,abp[start:stop],color='slategray')
            axes[row,1].scatter((bp-start)/FS,abp[bp],color='crimson',s=20)
            axes[row,1].set(title=f'ABP {len(bp)} peaks; rates PPG/ABP {s.ppg_rate_bpm:.1f}/{s.abp_rate_bpm:.1f} BPM',ylabel='ABP mmHg')
            for ax in axes[row]:ax.set_xlabel('Seconds');ax.grid(alpha=.2)
        save(fig,f'beat_review_{page+1}.png')
    print('Saved expanded EDA plots')


def rf_plots():
    metrics=pd.read_csv(RF/'validation_model_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,target in zip(axes,['sbp','dbp']):
        rows=metrics[metrics.target.eq(target)]
        ax.bar(rows.model,rows.mae_mmHg,color=COLORS)
        ax.set(title=f'{target.upper()} validation error',ylabel='MAE (mmHg)')
        ax.tick_params(axis='x',rotation=15)
    save(fig,'validation_model_comparison.png',RF)
    predictions=pd.read_csv(RF/'validation_predictions.csv')
    selected=predictions[predictions.model.eq('Random Forest')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,target in zip(axes,['sbp','dbp']):
        for i,(part,group) in enumerate(selected.groupby('part')):
            ax.scatter(group[target],group['prediction_'+target],s=12,alpha=.5,color=COLORS[i],label=part)
        low=min(selected[target].min(),selected['prediction_'+target].min())-3
        high=max(selected[target].max(),selected['prediction_'+target].max())+3
        ax.plot([low,high],[low,high],'k:')
        ax.set(title=target.upper(),xlabel='Draft target (mmHg)',ylabel='Random Forest prediction (mmHg)',xlim=(low,high),ylim=(low,high))
        ax.legend(fontsize=8)
    save(fig,'random_forest_predictions.png',RF)
    importance=pd.read_csv(RF/'random_forest_feature_importance.csv').head(12).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(importance.feature,importance.validation_permutation_mae_increase,
            xerr=importance.permutation_repeat_sd,color='teal')
    ax.axvline(0,color='gray',linewidth=.8)
    ax.set(xlabel='Increase in mean SBP/DBP validation MAE after shuffling (mmHg)',title='RF validation permutation importance; descriptive, not causal')
    save(fig,'random_forest_feature_importance.png',RF)
    print('Saved RF validation plots')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rf-only',action='store_true')
    args=parser.parse_args()
    if not args.rf_only:eda_plots()
    if (RF/'validation_model_comparison.csv').exists():rf_plots()
