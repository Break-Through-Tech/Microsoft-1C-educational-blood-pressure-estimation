"""Create an executed walkthrough of the saved RF refinement results."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    nb=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('''# Refining the expanded Random Forest

The expanded RF still predicts too close to the middle of the pressure range. We tested four focused changes: deeper shared trees, separate forests for the two pressures, equal-group training weights on the reference model, and equal-group weights on separate deeper forests.

The 60 features, 1,308 training recordings, quality rules, labels and grouped folds stay fixed. Selection uses training OOF predictions only. A replacement must improve mean target/group MAE by at least 0.05 mmHg without worsening either target. Only the chosen replacement and reference are scored on exploratory validation. The old test is not accessed.'''),
        nbformat.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
RUN=Path('outputs/rf_refinement_oct28')
decision=json.loads((RUN/'selection.json').read_text())
print('Training-selected candidate:', decision['selected'])
metrics=pd.read_csv(RUN/'training_oof_metrics.csv')
display(metrics.pivot(index='model',columns='target',values=['mae_mmHg','overlap_group_macro_mae']).round(3))'''),
        nbformat.v4.new_markdown_cell('''## Check validation after selecting

These are the same 857 validation windows used earlier, so the results are development evidence rather than a new final score. Lower MAE and RMSE are better. The paired intervals give each overlap group equal weight and do not account for all repeated model-development decisions.'''),
        nbformat.v4.new_code_cell("validation=pd.read_csv(RUN/'validation_metrics.csv')\ndisplay(validation.pivot(index='model',columns='target',values=['mae_mmHg','rmse_mmHg']).round(3))\ndisplay(pd.read_csv(RUN/'validation_paired_group_gains.csv').round(3))"),
        nbformat.v4.new_markdown_cell('''## Are extreme pressures still difficult?

Each point represents one training pressure quarter, using out-of-fold predictions. A good fit would stay near the diagonal. Predictions that cluster near the middle still miss low and high pressures, even if overall MAE improves.'''),
        nbformat.v4.new_code_cell("display(Image(filename=str(RUN/'training_shrinkage.png')))\nbands=pd.read_csv(RUN/'training_pressure_quartile_errors.csv')\ndisplay(bands[bands.model.isin(['Expanded reference RF',decision['selected']])].round(2))"),
        nbformat.v4.new_markdown_cell('''## Use the trained model on PPG

`predict_ppg_rf.py` takes a one-dimensional NumPy PPG recording at 125 Hz and a start sample. It extracts the same original and complete-pulse features, then loads an explicitly specified saved model. The requested five-second window needs at least one second of recording context on each side. PPG checks are applied; ABP-based label checks are unavailable at inference.

The raw-PPG prediction was checked against the saved-feature prediction on a training example. Model binaries stay local and must be regenerated before using an uploaded copy. These remain research predictions with provisional pressure labels and no confirmed patient-level evaluation. See [the full report](RF_REFINEMENT_RESULTS.md).'''),
        nbformat.v4.new_code_cell("display(pd.read_json(RUN/'inference_check.json',typ='series'))"),
        nbformat.v4.new_markdown_cell('''## Reproduce

`refine_expanded_rf.py --output-dir outputs/rf_refinement_new_run` reruns the experiment using the saved feature tables and earlier reference checkpoint. `summarize_rf_refinement.py` regenerates the report and diagnostics. Run `python -m unittest test_refine_expanded_rf test_predict_ppg_rf -v` for the weighting and inference-input checks.''')
    ])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}}).execute()
    nbformat.write(nb,'rf_refinement.ipynb')

if __name__=='__main__':main()
