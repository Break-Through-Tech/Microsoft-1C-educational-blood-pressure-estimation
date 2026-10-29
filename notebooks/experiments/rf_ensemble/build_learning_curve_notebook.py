"""Render the saved learning-curve results into a readable notebook."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    nb=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('''# Does training on more recordings help?

This experiment keeps the original validation windows, quality gates and pressure-label method fixed. It adds training recordings sampled evenly across Parts 1–4 without selecting on pressure values. All additions are screened for known exact/shifted overlaps with held-out recordings before fitting.

We compare the original features, normalized features and original-plus-pulse features using the fixed RF and boosting settings from the previous experiment. Validation is exploratory; the old test is not scored. [Full results](TRAINING_EXPANSION_RESULTS.md).'''),
        nbformat.v4.new_code_cell('''from pathlib import Path
import pandas as pd
from IPython.display import display, Image
RUN=Path('outputs/training_learning_curve_oct28')
metrics=pd.read_csv(RUN/'metrics.csv')
display(pd.read_csv(RUN/'training_counts.csv'))
validation=metrics[metrics.split.eq('validation')]
display(validation.pivot(index=['level','variant','model'],columns='target',values='mae_mmHg').round(2))'''),
        nbformat.v4.new_markdown_cell('''## Learning curve

The horizontal axis counts distinct training recordings. The vertical axis is validation MAE in mmHg, so lower is better. Every point uses the same validation windows. A lower training/OOF error alone would not answer this question because the training population changes as recordings are added.'''),
        nbformat.v4.new_code_cell("display(Image(filename=str(RUN/'learning_curve.png')))"),
        nbformat.v4.new_markdown_cell('''## Check large errors and equal group weights

RMSE gives more weight to large misses. Equal-group MAE gives each detected overlap group the same weight instead of allowing recordings with more windows to count more. Positive paired gains mean lower equal-group MAE than the original-size model using the same features. Intervals are conditional on these fixed models and do not include all model-development uncertainty.'''),
        nbformat.v4.new_code_cell("display(validation[['level','variant','model','target','rmse_mmHg','overlap_group_macro_mae']].round(2))\ndisplay(pd.read_csv(RUN/'paired_expansion_gains.csv').round(3))"),
        nbformat.v4.new_markdown_cell('''## Training-only checks

The table below evaluates out-of-fold predictions on the same original 4,013 training windows at every size. Folds are defined by overlap groups on the complete training pool and restricted to each nested training subset; the original evaluation groups stay held out inside each fold. Each model and feature condition at the same size uses the same folds. Original test records and connected training candidates are excluded from model fitting. The dataset has recording IDs, not confirmed patient IDs.'''),
        nbformat.v4.new_code_cell("display(pd.read_csv(RUN/'fixed_original_oof_metrics.csv')[['level','variant','model','target','mae_mmHg','overlap_group_macro_mae']].round(2))\ndisplay(pd.read_json('outputs/training_expansion_oct28/preparation_summary.json',typ='series'))"),
        nbformat.v4.new_markdown_cell('''## Reproduce

Run `expand_training_recordings.py` to prepare the additions, then `training_recording_learning_curve.py` to extract features and fit the fixed models. Both require new output directories for a new run. `summarize_training_learning_curve.py` regenerates the comparisons, chart and report from saved results. Raw MAT data and generated model binaries are not included in the upload.

A future final score needs a new untouched evaluation set. The fixed validation results here remain useful for development, but repeated inspection makes them an optimistic guide to future performance.''')
    ])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}}).execute()
    nbformat.write(nb,'training_recording_learning_curve.ipynb')

if __name__=='__main__':main()
