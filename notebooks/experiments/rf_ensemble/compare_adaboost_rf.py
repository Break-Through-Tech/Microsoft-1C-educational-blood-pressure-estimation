"""AdaBoost and RF refinements on the existing PPG feature table.

Select settings on three overlap-grouped training folds. Report validation as
exploratory: it has been inspected before. Never predict on the old test split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from random_forest_pipeline import load_data, metric_rows, group_bootstrap, TARGETS

SEED = 20261028


def wrapped(model):
    return TransformedTargetRegressor(regressor=Pipeline([
        ('imputer', SimpleImputer(strategy='median')), ('model', model)]),
        transformer=StandardScaler())


def main(data, original, output):
    output.mkdir(parents=True, exist_ok=False)
    table, features = load_data(data)
    settings = json.loads((original / 'model_settings.json').read_text())
    for name, expected in settings['data_sha256'].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Dataset changed: ' + name)
    train = table[table.eligible & table.split.eq('train')].copy()
    validation = table[table.eligible & table.split.eq('validation')].copy()
    X = train[features].replace([np.inf, -np.inf], np.nan)
    y = train[TARGETS].to_numpy()
    folds = list(GroupKFold(3).split(X, y, train.overlap_group))
    assignments = train[['window_key', 'record_key', 'overlap_group']].copy()
    assignments['fold'] = -1
    for i, (inside, held) in enumerate(folds):
        assert set(train.overlap_group.iloc[inside]).isdisjoint(train.overlap_group.iloc[held])
        assignments.iloc[held, assignments.columns.get_loc('fold')] = i
    assignments.to_csv(output / 'training_fold_assignments.csv', index=False)
    rf = RandomForestRegressor(n_estimators=200, max_features=.5,
        max_samples=.6, n_jobs=2, random_state=SEED)
    separate = MultiOutputRegressor(RandomForestRegressor(n_estimators=200,
        max_features=.5, max_samples=.7, n_jobs=2, random_state=SEED), n_jobs=1)
    ada = MultiOutputRegressor(AdaBoostRegressor(
        estimator=DecisionTreeRegressor(min_samples_leaf=10, random_state=SEED),
        loss='linear', random_state=SEED), n_jobs=1)
    candidates = {
        'AdaBoost': (wrapped(ada), {
            'regressor__model__estimator__estimator__max_depth': [2, 4],
            'regressor__model__estimator__learning_rate': [.05, .3],
            'regressor__model__estimator__n_estimators': [50, 150]}),
        'Regularized RF': (wrapped(rf), {
            'regressor__model__max_depth': [4, 8],
            'regressor__model__min_samples_leaf': [10, 25]}),
        'Separate-target RF': (wrapped(separate), {
            'regressor__model__estimator__max_depth': [8, None],
            'regressor__model__estimator__min_samples_leaf': [5, 15]}),
    }
    protocol = dict(seed=SEED, sklearn=sklearn.__version__, features=features,
        data_sha256=settings['data_sha256'], training_windows=len(train),
        validation_windows=len(validation),
        selection='Mean window SBP/DBP MAE in mmHg; three training-only overlap-group folds',
        validation='Exploratory report only; previously inspected in earlier experiments',
        test_evaluated=False, old_test_status='Previously evaluated; not reused here',
        grids={name: grid for name, (_, grid) in candidates.items()},
        fixed_estimators={name: repr(estimator) for name, (estimator, _) in candidates.items()},
        note='Best-setting CV scores are selection scores, not unbiased final performance estimates')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    models = {
        'Training mean': joblib.load(original / 'training_mean.joblib'),
        'Training median': DummyRegressor(strategy='median').fit(X, y),
        'Original RF': joblib.load(original / 'random_forest.joblib'),
        'Original Extra Trees': joblib.load(original / 'extra_trees.joblib'),
    }
    chosen = {}
    for name, (estimator, grid) in candidates.items():
        print('Grouped training search:', name, flush=True)
        search = GridSearchCV(estimator, grid, cv=folds,
            scoring={'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'},
            refit='mae', return_train_score=True, error_score='raise', n_jobs=1)
        with threadpool_limits(limits=2):
            search.fit(X, y)
        slug = name.lower().replace(' ', '_').replace('-', '_')
        pd.DataFrame(search.cv_results_).to_csv(output / (slug + '_cv.csv'), index=False)
        models[name] = search.best_estimator_
        joblib.dump(search.best_estimator_, output / (slug + '.joblib'))
        chosen[name] = dict(params=search.best_params_, selection_cv_mae=-search.best_score_)
        print(name, chosen[name], flush=True)
    (output / 'selected_settings.json').write_text(json.dumps(chosen, indent=2))
    for split, subset in [('train', train), ('validation', validation)]:
        values = subset[features].replace([np.inf, -np.inf], np.nan)
        truth = subset[TARGETS].to_numpy()
        rows, frames, parts = [], [], []
        for name, model in models.items():
            with threadpool_limits(limits=2):
                prediction = model.predict(values)
            assert prediction.shape == truth.shape and np.isfinite(prediction).all()
            rows.extend(metric_rows(subset, truth, prediction, name))
            frame = subset[['part', 'record_key', 'window_key', 'overlap_group']].copy()
            frame['model'] = name
            for i, target in enumerate(TARGETS):
                frame[target] = truth[:, i]
                frame['prediction_' + target] = prediction[:, i]
            frames.append(frame)
            for part in sorted(subset.part.unique()):
                mask = subset.part.eq(part).to_numpy()
                parts.extend(dict(r, part=part) for r in metric_rows(
                    subset.loc[mask], truth[mask], prediction[mask], name))
        predictions, metrics = pd.concat(frames, ignore_index=True), pd.DataFrame(rows)
        predictions.to_csv(output / f'{split}_predictions.csv', index=False)
        metrics.to_csv(output / f'{split}_metrics.csv', index=False)
        if split == 'validation':
            pd.DataFrame(parts).to_csv(output / 'validation_by_part.csv', index=False)
            group_bootstrap(predictions, seed=SEED, repeats=2000).to_csv(
                output / 'validation_group_bootstrap.csv', index=False)
        print(split.upper(), flush=True)
        print(metrics[['model', 'target', 'mae_mmHg', 'rmse_mmHg']].round(3).to_string(index=False), flush=True)
    (output / 'COMPLETED.json').write_text(json.dumps(dict(status='complete', test_evaluated=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('outputs/expanded_eda_oct14'))
    parser.add_argument('--original-dir', type=Path, default=Path('outputs/random_forest_oct14'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/adaboost_rf_oct28'))
    args = parser.parse_args()
    main(args.data_dir, args.original_dir, args.output_dir)
