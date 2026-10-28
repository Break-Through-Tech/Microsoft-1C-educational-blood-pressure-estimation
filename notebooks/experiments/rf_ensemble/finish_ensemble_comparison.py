"""Finish a frozen RF/ensemble comparison; score the reserved test once.

Run with OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python finish_ensemble_comparison.py
All model selection uses training overlap-group folds. Test results never select
settings or ensemble weights. Existing RF/Extra Trees checkpoints stay frozen.
"""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from random_forest_pipeline import load_data, metric_rows, group_bootstrap, TARGETS

DATA = Path('outputs/expanded_eda_oct14')
OLD = Path('outputs/random_forest_oct14')
OUT = Path('outputs/ensemble_final_oct28')


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    table, features = load_data(DATA)
    old_settings = json.loads((OLD / 'model_settings.json').read_text())
    for name, expected in old_settings['data_sha256'].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected, name
    train = table[table.eligible & table.split.eq('train')]
    X = train[features].replace([np.inf, -np.inf], np.nan)
    y = train[TARGETS].to_numpy()
    folds = list(GroupKFold(3).split(X, y, train.overlap_group))
    protocol = dict(
        date='2026-10-28', features=features, data_sha256=old_settings['data_sha256'],
        models=['Training mean', 'Random Forest', 'Extra Trees', 'Histogram boosting', 'Equal-weight ensemble'],
        frozen_forests='Original October 14 training-only checkpoints; no refit on validation',
        boosting_selection='Four configurations, three grouped training folds; equal-weight target MAE in mmHg',
        boosting_grid={'max_leaf_nodes': [7, 15], 'l2_regularization': [1.0, 10.0]},
        boosting_fixed=dict(max_iter=150, learning_rate=.05, min_samples_leaf=20, early_stopping=False),
        ensemble='Arithmetic average of RF, Extra Trees and boosting; fixed one-third weights',
        test_policy='Report all frozen models; no test-based model or weight selection',
        limits='Provisional labels/quality gates; recording/overlap-group split; patient independence unknown')
    # Written before fitting or generating any test predictions.
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    estimator = TransformedTargetRegressor(regressor=MultiOutputRegressor(
        HistGradientBoostingRegressor(random_state=20261028, **protocol['boosting_fixed']), n_jobs=1),
        transformer=StandardScaler())
    search = GridSearchCV(estimator, {
        'regressor__estimator__max_leaf_nodes': [7, 15],
        'regressor__estimator__l2_regularization': [1., 10.]},
        cv=folds, scoring='neg_mean_absolute_error', n_jobs=1, error_score='raise')
    print('Fitting boosting: four configurations x three grouped folds.', flush=True)
    with threadpool_limits(limits=2):
        search.fit(X, y)
    pd.DataFrame(search.cv_results_).to_csv(OUT / 'boosting_grouped_cv.csv', index=False)
    models = {name: joblib.load(OLD / filename) for name, filename in [
        ('Training mean', 'training_mean.joblib'), ('Random Forest', 'random_forest.joblib'),
        ('Extra Trees', 'extra_trees.joblib')]}
    models['Histogram boosting'] = search.best_estimator_
    joblib.dump(search.best_estimator_, OUT / 'histogram_boosting.joblib')
    (OUT / 'selection.json').write_text(json.dumps(dict(best_params=search.best_params_,
        grouped_training_cv_mae=-search.best_score_), indent=2))
    for split in ['validation', 'test']:
        subset = table[table.eligible & table.split.eq(split)]
        values = subset[features].replace([np.inf, -np.inf], np.nan)
        truth = subset[TARGETS].to_numpy()
        with threadpool_limits(limits=2):
            guesses = {name: model.predict(values) for name, model in models.items()}
        guesses['Equal-weight ensemble'] = np.mean([guesses[n] for n in
            ['Random Forest', 'Extra Trees', 'Histogram boosting']], axis=0)
        metrics, frames, parts = [], [], []
        for name, prediction in guesses.items():
            assert prediction.shape == truth.shape and np.isfinite(prediction).all()
            metrics.extend(metric_rows(subset, truth, prediction, name))
            frame = subset[['part', 'record_key', 'window_key', 'overlap_group']].copy()
            frame['model'] = name
            for i, target in enumerate(TARGETS):
                frame[target] = truth[:, i]
                frame['prediction_' + target] = prediction[:, i]
            frames.append(frame)
            for part in sorted(subset.part.unique()):
                mask = subset.part.eq(part).to_numpy()
                parts.extend(dict(row, part=part) for row in metric_rows(
                    subset.loc[mask], truth[mask], prediction[mask], name))
        metrics = pd.DataFrame(metrics)
        predictions = pd.concat(frames, ignore_index=True)
        metrics.to_csv(OUT / f'{split}_metrics.csv', index=False)
        predictions.to_csv(OUT / f'{split}_predictions.csv', index=False)
        pd.DataFrame(parts).to_csv(OUT / f'{split}_metrics_by_part.csv', index=False)
        group_bootstrap(predictions, seed=20261028, repeats=2000).to_csv(
            OUT / f'{split}_group_bootstrap.csv', index=False)
        if split == 'validation':
            original = pd.read_csv(OLD / 'validation_model_comparison.csv')
            for name in models.keys() - {'Histogram boosting'}:
                for target in TARGETS:
                    previous = original.query('model == @name and target == @target').mae_mmHg.iloc[0]
                    current = metrics.query('model == @name and target == @target').mae_mmHg.iloc[0]
                    assert np.isclose(previous, current), (name, target)
        print(split.upper(), flush=True)
        print(metrics[['model', 'target', 'mae_mmHg', 'rmse_mmHg']].round(3).to_string(index=False), flush=True)
    (OUT / 'COMPLETED.json').write_text(json.dumps(dict(test_evaluated=True,
        status='Completed frozen comparison; test must not be reused for tuning'), indent=2))


if __name__ == '__main__':
    main()
