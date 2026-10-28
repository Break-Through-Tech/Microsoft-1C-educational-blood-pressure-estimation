"""Expanded Random Forest tuning using grouped training CV; validation is report-only."""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.model_selection import GridSearchCV, GroupKFold
from random_forest_pipeline import load_data, forest_model, metric_rows, TARGETS

DATA = Path('outputs/expanded_eda_oct14')
RUN = Path('outputs/random_forest_oct14')
OUT = Path('outputs/rf_tuning_oct14')
SEED = 20261014


def paired_group_bootstrap(old_predictions, new_predictions, repeats=1000):
    rng = np.random.default_rng(SEED)
    rows = []
    for target in TARGETS:
        old = old_predictions.groupby('overlap_group')[f'error_{target}'].apply(lambda x: np.abs(x).mean())
        new = new_predictions.groupby('overlap_group')[f'error_{target}'].apply(lambda x: np.abs(x).mean())
        groups = sorted(set(old.index) & set(new.index))
        old_values, new_values = old.reindex(groups).to_numpy(), new.reindex(groups).to_numpy()
        # Positive means the expanded search lowered group-macro MAE.
        gains = old_values - new_values
        sampled = rng.integers(0, len(groups), size=(repeats, len(groups)))
        samples = gains[sampled].mean(axis=1)
        rows.append(dict(target=target, overlap_groups=len(groups),
            original_group_macro_mae=float(old_values.mean()),
            expanded_tune_group_macro_mae=float(new_values.mean()),
            gain_mmHg=float(gains.mean()), gain_ci_low=float(np.quantile(samples,.025)),
            gain_ci_high=float(np.quantile(samples,.975))))
    return pd.DataFrame(rows)


def main(output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    table, features = load_data(DATA)
    train = table[table.split.eq('train')]
    validation = table[table.split.eq('validation')]
    X = train[features].replace([np.inf,-np.inf],np.nan)
    y = train[TARGETS].to_numpy()
    V = validation[features].replace([np.inf,-np.inf],np.nan)
    vy = validation[TARGETS].to_numpy()
    groups = train.overlap_group.to_numpy()
    folds = list(GroupKFold(n_splits=3).split(X,y,groups))
    for inner, held in folds:
        assert set(groups[inner]).isdisjoint(groups[held])
    grid = {
        'regressor__forest__max_depth':[8,14,None],
        'regressor__forest__min_samples_leaf':[2,5,10],
        'regressor__forest__max_features':[.5,.8,1.0],
    }
    search = GridSearchCV(forest_model(seed=SEED), grid, cv=folds,
        scoring='neg_mean_absolute_error', n_jobs=1, return_train_score=True,
        error_score='raise')
    print(f'Expanded grouped CV: {len(list(__import__("sklearn").model_selection.ParameterGrid(grid)))} settings x 3 folds.',flush=True)
    search.fit(X,y)
    pd.DataFrame(search.cv_results_).to_csv(output/'grouped_training_cv.csv',index=False)
    prediction = search.best_estimator_.predict(V)
    metrics = pd.DataFrame(metric_rows(validation,vy,prediction,'Expanded-tuned RF'))
    metrics.to_csv(output/'validation_metrics.csv',index=False)
    frame = validation[['part','record_id','record_key','window_key','overlap_group']].copy()
    for i,target in enumerate(TARGETS):
        frame[target]=vy[:,i]
        frame['prediction_'+target]=prediction[:,i]
        frame['error_'+target]=prediction[:,i]-vy[:,i]
    frame.to_csv(output/'validation_predictions.csv',index=False)
    joblib.dump(search.best_estimator_,output/'expanded_tuned_random_forest.joblib')
    original=pd.read_csv(RUN/'validation_predictions.csv').query("model == 'Random Forest'")
    paired_group_bootstrap(original,frame).to_csv(output/'paired_group_bootstrap_vs_original_rf.csv',index=False)
    comparison=pd.concat([pd.read_csv(RUN/'validation_model_comparison.csv'),metrics],ignore_index=True)
    comparison.to_csv(output/'validation_comparison.csv',index=False)
    settings=dict(seed=SEED,sklearn=sklearn.__version__,features=features,
        training_windows=len(train),training_overlap_groups=int(train.overlap_group.nunique()),
        validation_windows=len(validation),validation_overlap_groups=int(validation.overlap_group.nunique()),
        reserved_test_windows=int(table.split.eq('test').sum()),test_evaluated=False,
        selection='27 configurations, three-fold GroupKFold by detected overlap group; mean multi-output MAE',
        grid=grid,best_params=search.best_params_,grouped_training_cv_mae=float(-search.best_score_),
        n_estimators=200,seed_forest=SEED)
    (output/'settings.json').write_text(json.dumps(settings,indent=2))
    print('Best settings:',search.best_params_,'grouped training CV MAE:',round(-search.best_score_,3))
    print(comparison[['model','target','mae_mmHg','rmse_mmHg','bias_mmHg','recording_macro_mae']].round(2).to_string(index=False))
    print('\nPaired group bootstrap against original RF:')
    print(pd.read_csv(output/'paired_group_bootstrap_vs_original_rf.csv').round(3).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    main(args.output_dir)
