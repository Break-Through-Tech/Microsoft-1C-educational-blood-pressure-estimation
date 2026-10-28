"""Descriptive error analysis for the saved RF validation predictions only."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('outputs/random_forest_oct14')
PRED = ROOT / 'validation_predictions.csv'


def main():
    predictions = pd.read_csv(PRED)
    # Windows are repeated within recordings; every table reports both window and
    # recording counts and is descriptive, not an independent-subject estimate.
    rows = []
    for target, edges, labels in [
        ('sbp', [-float('inf'), 100, 120, 140, 160, float('inf')],
         ['<100', '100-119', '120-139', '140-159', '160+']),
        ('dbp', [-float('inf'), 60, 70, 80, 90, float('inf')],
         ['<60', '60-69', '70-79', '80-89', '90+']),
    ]:
        for model, data in predictions.groupby('model'):
            data = data.copy()
            data['pressure_band'] = pd.cut(data[target], edges, labels=labels, right=False)
            for band, group in data.groupby('pressure_band', observed=True):
                error = group[f'prediction_{target}'] - group[target]
                rows.append(dict(target=target.upper(), pressure_band=str(band), model=model,
                    windows=len(group), recordings=group.record_key.nunique(),
                    mean_reference_mmHg=group[target].mean(), mean_prediction_mmHg=group[f'prediction_{target}'].mean(),
                    mae_mmHg=error.abs().mean(), rmse_mmHg=(error.pow(2).mean())**.5,
                    bias_mmHg=error.mean()))
    by_band = pd.DataFrame(rows)
    by_band.to_csv(ROOT / 'validation_error_by_pressure_band.csv', index=False)

    rf = predictions[predictions.model.eq('Random Forest')].copy()
    part_rows = []
    for part in sorted(rf.part.unique()):
        g = rf[rf.part.eq(part)]
        for target in ['sbp', 'dbp']:
            err = g[f'error_{target}']
            part_rows.append(dict(part=part, target=target.upper(), windows=len(g),
                recordings=g.record_key.nunique(), mae_mmHg=err.abs().mean(),
                rmse_mmHg=(err.pow(2).mean())**.5, bias_mmHg=err.mean()))
    by_part = pd.DataFrame(part_rows)
    by_part.to_csv(ROOT / 'validation_error_by_part_rf.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'Training mean':'#9b9b9b','Ridge':'#2673a8','Random Forest':'#d05a37','Extra Trees':'#6c8f45'}
    for ax, target in zip(axes, ['SBP','DBP']):
        view = by_band[by_band.target.eq(target)]
        bands = list(view.pressure_band.drop_duplicates())
        x = range(len(bands))
        width = .19
        for j, model in enumerate(['Training mean','Ridge','Random Forest','Extra Trees']):
            z = view[view.model.eq(model)].set_index('pressure_band').reindex(bands)
            ax.bar([v + (j-1.5)*width for v in x], z.mae_mmHg, width=width, label=model, color=colors[model])
        ax.set_xticks(list(x), bands)
        ax.set_xlabel(f'Reference {target} (mmHg)')
        ax.set_ylabel('Validation MAE (mmHg)')
        ax.set_title(f'{target} error across reference ranges')
        ax.grid(axis='y', alpha=.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Error by pressure range (validation windows; repeated windows per recording)')
    fig.tight_layout()
    fig.savefig(ROOT / 'validation_error_by_pressure_band.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target in zip(axes, ['sbp','dbp']):
        z = rf
        ax.scatter(z[target], z[f'prediction_{target}'], s=13, alpha=.35, color='#2673a8', edgecolors='none')
        low = min(z[target].min(), z[f'prediction_{target}'].min())
        high = max(z[target].max(), z[f'prediction_{target}'].max())
        ax.plot([low, high], [low, high], color='#b54b35', linestyle='--', linewidth=1)
        ax.set_xlabel(f'Reference {target.upper()} (mmHg)')
        ax.set_ylabel(f'Random Forest prediction (mmHg)')
        ax.set_title(f'{target.upper()} predictions vs reference')
        ax.grid(alpha=.2)
    fig.suptitle('Random Forest validation predictions')
    fig.tight_layout()
    fig.savefig(ROOT / 'validation_rf_predictions_scatter.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(by_band[by_band.model.isin(['Training mean','Random Forest'])].round(2).to_string(index=False))
    print('\nSaved validation-only pressure-band and Part error tables and figures.')


if __name__ == '__main__':
    main()
