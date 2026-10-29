"""PPG-feature models on the expanded all-Part, overlap-grouped data.

Tuning uses grouped training folds; reporting uses validation only. Test stays
reserved. ABP, IDs, quality gates and diagnostic count comparisons are not inputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ['sbp', 'dbp']


def load_data(directory):
    directory = Path(directory)
    table = pd.read_csv(directory / 'expanded_features_and_split.csv', low_memory=False)
    features = json.loads((directory / 'feature_columns.json').read_text())
    if not features or any(not name.startswith(('ppg_', 'vpg_', 'apg_')) for name in features):
        raise ValueError('Feature whitelist contains a non-PPG predictor')
    forbidden = {'ppg_quality_mean', 'ppg_peak_count', 'ppg_count_bpm', 'ppg_rate_bpm'}
    if forbidden & set(features):
        raise ValueError('Quality/diagnostic metadata entered model features')
    eligible = table[table.eligible]
    if not eligible.split.isin(['train','validation','test']).all():
        raise ValueError('Unknown split')
    for identity in ['record_key', 'overlap_group', 'raw_ppg_sha256']:
        if eligible.groupby(identity).split.nunique().max() != 1:
            raise ValueError(f'{identity} crosses splits')
    if not np.isfinite(eligible[TARGETS]).all().all():
        raise ValueError('Nonfinite eligible targets')
    return table, features


def forest_model(extra=False, seed=20261014):
    Forest = ExtraTreesRegressor if extra else RandomForestRegressor
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('forest', Forest(n_estimators=200, min_samples_leaf=5,
            max_features=.8, random_state=seed, n_jobs=2)),
    ])
    # Equalize target scales when the two outputs share tree splits. Fitted
    # inside every training fold; predict() automatically returns mmHg.
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def metric_rows(metadata, truth, guesses, model):
    rows = []
    for i, target in enumerate(TARGETS):
        error = guesses[:, i] - truth[:, i]
        groups = metadata[['record_key','overlap_group']].copy()
        groups['absolute_error'] = np.abs(error)
        rows.append(dict(model=model, target=target, windows=len(error),
            records=metadata.record_key.nunique(), overlap_groups=metadata.overlap_group.nunique(),
            mae_mmHg=float(np.mean(np.abs(error))), rmse_mmHg=float(np.sqrt(np.mean(error ** 2))),
            bias_mmHg=float(np.mean(error)), error_sd_mmHg=float(np.std(error, ddof=1)),
            recording_macro_mae=float(groups.groupby('record_key').absolute_error.mean().mean()),
            overlap_group_macro_mae=float(groups.groupby('overlap_group').absolute_error.mean().mean())))
    return rows


def group_bootstrap(predictions, seed=20261014, repeats=500):
    rng = np.random.default_rng(seed)
    baseline = predictions[predictions.model.eq('Training mean')]
    groups = sorted(baseline.overlap_group.unique())
    rows = []
    for target in TARGETS:
        base = baseline.assign(ae=(baseline['prediction_'+target]-baseline[target]).abs()).groupby('overlap_group').ae.mean().reindex(groups).to_numpy()
        indices = rng.integers(0,len(groups),size=(repeats,len(groups)))
        for name, values in predictions.groupby('model'):
            ae = values.assign(ae=(values['prediction_'+target]-values[target]).abs()).groupby('overlap_group').ae.mean().reindex(groups).to_numpy()
            means = ae[indices].mean(axis=1)
            improvement = (base[indices]-ae[indices]).mean(axis=1)
            rows.append(dict(model=name,target=target,overlap_groups=len(groups),
                group_macro_mae=float(ae.mean()),mae_ci_low=float(np.quantile(means,.025)),
                mae_ci_high=float(np.quantile(means,.975)),
                gain_over_mean_baseline=float((base-ae).mean()),
                gain_ci_low=float(np.quantile(improvement,.025)),gain_ci_high=float(np.quantile(improvement,.975))))
    return pd.DataFrame(rows)


def train(data_dir=Path('outputs/expanded_eda_oct14'), output=Path('outputs/random_forest_oct14'), seed=20261014):
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    table, features = load_data(data_dir)
    training = table[table.split.eq('train')].copy()
    validation = table[table.split.eq('validation')].copy()
    X = training[features].replace([np.inf,-np.inf],np.nan)
    y = training[TARGETS].to_numpy()
    V = validation[features].replace([np.inf,-np.inf],np.nan)
    vy = validation[TARGETS].to_numpy()
    if X.notna().sum().eq(0).any():
        raise ValueError('A predefined feature is completely missing in training')
    groups = training.overlap_group.to_numpy()
    folds = list(GroupKFold(n_splits=3).split(X,y,groups))
    assignments = training[['part','record_id','window_key','overlap_group']].copy()
    assignments['cv_validation_fold'] = -1
    for fold,(inside,held) in enumerate(folds):
        assert set(groups[inside]).isdisjoint(set(groups[held]))
        assignments.iloc[held,assignments.columns.get_loc('cv_validation_fold')] = fold
    assignments.to_csv(output / 'training_grouped_cv_assignments.csv',index=False)
    search = GridSearchCV(forest_model(seed=seed),
        param_grid={'regressor__forest__max_depth':[10,None],
                    'regressor__forest__min_samples_leaf':[5,10]},
        cv=folds,scoring='neg_mean_absolute_error',n_jobs=1,return_train_score=True,error_score='raise')
    print(f'Fitting RF: {len(training)} training windows, {len(features)} features, four configurations and three grouped folds',flush=True)
    search.fit(X,y)
    pd.DataFrame(search.cv_results_).to_csv(output / 'random_forest_cv_results.csv',index=False)
    models = {
        'Training mean':DummyRegressor(strategy='mean').fit(X,y),
        'Ridge':Pipeline([('imputer',SimpleImputer(strategy='median')),
            ('scale',StandardScaler()),('ridge',Ridge(alpha=10))]).fit(X,y),
        'Random Forest':search.best_estimator_,
        'Extra Trees':forest_model(extra=True,seed=seed).fit(X,y),
    }
    metrics, predictions, part_metrics = [], [], []
    for name, model in models.items():
        guesses = model.predict(V)
        if not np.isfinite(guesses).all():
            raise RuntimeError(f'Nonfinite predictions: {name}')
        metrics.extend(metric_rows(validation,vy,guesses,name))
        result = validation[['part','record_id','record_key','window_key','overlap_group','ppg_quality_mean']].copy()
        result['model'] = name
        for i,target in enumerate(TARGETS):
            result[target],result['prediction_'+target] = vy[:,i],guesses[:,i]
            result['error_'+target] = guesses[:,i]-vy[:,i]
        predictions.append(result)
        for part in sorted(validation.part.unique()):
            mask = validation.part.eq(part).to_numpy()
            for r in metric_rows(validation.loc[mask],vy[mask],guesses[mask],name):
                r['part'] = part
                part_metrics.append(r)
        joblib.dump(model,output / (name.lower().replace(' ','_')+'.joblib'))
    predicted = pd.concat(predictions,ignore_index=True)
    metric_table = pd.DataFrame(metrics)
    metric_table.to_csv(output / 'validation_model_comparison.csv',index=False)
    pd.DataFrame(part_metrics).to_csv(output / 'validation_metrics_by_part.csv',index=False)
    predicted.to_csv(output / 'validation_predictions.csv',index=False)
    record_errors = predicted.groupby(['model','part','record_key','overlap_group'])[['error_sbp','error_dbp']].agg(lambda e:np.mean(np.abs(e)))
    record_errors.to_csv(output / 'validation_record_mae.csv')
    group_bootstrap(predicted,seed).to_csv(output / 'validation_group_bootstrap.csv',index=False)
    print('Computing descriptive validation permutation importance',flush=True)
    importance = permutation_importance(models['Random Forest'],V,vy,
        scoring='neg_mean_absolute_error',n_repeats=5,random_state=seed,n_jobs=2)
    forest = models['Random Forest'].regressor_.named_steps['forest']
    pd.DataFrame(dict(feature=features,validation_permutation_mae_increase=importance.importances_mean,
        permutation_repeat_sd=importance.importances_std,training_impurity_importance=forest.feature_importances_)).sort_values(
        'validation_permutation_mae_increase',ascending=False).to_csv(output / 'random_forest_feature_importance.csv',index=False)
    config = dict(seed=seed,sklearn=sklearn.__version__,feature_columns=features,targets=TARGETS,
        training_windows=len(training),training_records=int(training.record_key.nunique()),
        validation_windows=len(validation),validation_records=int(validation.record_key.nunique()),
        test_windows_reserved=int(table.split.eq('test').sum()),test_evaluated=False,
        rf_best_params=search.best_params_,rf_grouped_cv_mae=float(-search.best_score_),
        rf_selection='Mean SBP/DBP MAE over three grouped training folds only',
        rf_n_estimators=200,rf_max_features=.8,forest_targets='Standardized inside each training fold',
        ridge_alpha=10,extra_trees='Predefined comparison; not separately tuned',
        imputation='Training-only median via Pipeline; scaler training-only via target wrapper',
        data_status='Provisional quality/beat gates; no confirmed patient identifiers',
        importance_scope='Validation permutation (five repeats); correlated features can share importance; not causal or uncertainty-calibrated',
        bootstrap_scope='500 paired overlap-group resamples; group-macro MAE; patient independence not established',
        data_sha256={name:hashlib.sha256((data_dir/name).read_bytes()).hexdigest() for name in
            ['expanded_features_and_split.csv','feature_columns.json','settings.json']})
    (output/'model_settings.json').write_text(json.dumps(config,indent=2))
    print(metric_table[['model','target','mae_mmHg','rmse_mmHg','recording_macro_mae']].round(3).to_string(index=False),flush=True)
    return metric_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('outputs/expanded_eda_oct14'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/random_forest_oct14'))
    parser.add_argument('--seed',type=int,default=20261014)
    args = parser.parse_args()
    train(args.data_dir,args.output_dir,args.seed)
