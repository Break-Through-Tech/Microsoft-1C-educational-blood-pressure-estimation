"""Small presentation helpers; reference ABP is only used for display."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from demo_features import predict
BASE=Path(__file__).resolve().parent / 'assets'
examples=pd.read_csv(BASE/'examples.csv')

def show_example(example=1):
    row=examples.set_index('example').loc[example]
    with np.load(BASE/'signals.npz',allow_pickle=False) as data:
        ppg=data[f'ppg_{example}'];ecg=data[f'ecg_{example}'];abp=data[f'abp_{example}']
    start=int(row.start_sample)
    result=predict(ppg,ecg,start,BASE/'rf_ecg.joblib')
    t=np.arange(625)/125
    fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True)
    for ax,signal,label in zip(axes,[ppg[start:start+625],ecg[start:start+625],abp],['PPG (input)','ECG (input)','ABP (reference only, mmHg)']):
        ax.plot(t,signal,lw=1.2);ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (seconds)');fig.suptitle(f'Example {example}: {row.window_key}');fig.tight_layout();plt.show()
    table=pd.DataFrame({'Target':['Systolic','Diastolic'],'Reference (mmHg)':[row.sbp,row.dbp],
        'RF prediction (mmHg)':[result['sbp_mmHg'],result['dbp_mmHg']]})
    table['Absolute error (mmHg)']=abs(table['Reference (mmHg)']-table['RF prediction (mmHg)'])
    print(f"PPG template quality: {result['ppg_template_quality']:.2f}; PPG peaks: {result['ppg_peaks']}; ECG R peaks: {result['ecg_r_peaks']}")
    return table.round(2)
