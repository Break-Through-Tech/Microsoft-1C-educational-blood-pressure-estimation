"""Grouped training-CV selection and validation report for the RBF SVR."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from random_forest_pipeline import load_data, metric_rows, TARGETS

DATA = Path('outputs/expanded_eda_oct14')
RUN = Path('outputs/random_forest_oct14')
OUT = Path('outputs/svr_grouped_oct14')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    table, features = load_data(DATA)
    train = table[table.split.eq('train')]
    validation = table[table.split.eq('validation')]
    X = train[features].replace([np.inf, -np.inf], np.nan)
    y = train[TARGETS].to_numpy()
    V = validation[features].replace([np.inf, -np.inf], np.nan)
    vy = validation[TARGETS].to_numpy()
    groups = train.overlap_group.to_numpy()
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))
    pipeline = TransformedTargetRegressor(
        regressor=Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', MultiOutputRegressor(SVR(kernel='rbf'), n_jobs=1)),
        ]), transformer=StandardScaler())
    search = GridSearchCV(pipeline, {
        'regressor__model__estimator__C': [1, 10],
        'regressor__model__estimator__epsilon': [.1, .3],
    }, cv=folds, scoring='neg_mean_absolute_error', n_jobs=1,
       return_train_score=True, error_score='raise')
    print('Fitting RBF SVR settings on three grouped training folds.', flush=True)
    search.fit(X, y)
    pd.DataFrame(search.cv_results_).to_csv(OUT / 'grouped_training_cv.csv', index=False)
    predictions = search.best_estimator_.predict(V)
    metrics = pd.DataFrame(metric_rows(validation, vy, predictions, 'RBF SVR'))
    metrics.to_csv(OUT / 'validation_metrics.csv', index=False)
    frame = validation[['part','record_id','record_key','window_key','overlap_group']].copy()
    frame['model'] = 'RBF SVR'
    for i, target in enumerate(TARGETS):
        frame[target] = vy[:, i]
        frame['prediction_' + target] = predictions[:, i]
        frame['error_' + target] = predictions[:, i] - vy[:, i]
    frame.to_csv(OUT / 'validation_predictions.csv', index=False)
    joblib.dump(search.best_estimator_, OUT / 'rbf_svr.joblib')
    settings = {'sklearn': sklearn.__version__, 'features': features,
        'training_windows': len(train), 'training_overlap_groups': int(train.overlap_group.nunique()),
        'validation_windows': len(validation), 'validation_overlap_groups': int(validation.overlap_group.nunique()),
        'test_evaluated': False, 'selection': '3-fold GroupKFold; mean multi-output MAE',
        'best_params': search.best_params_, 'grouped_training_cv_mae': float(-search.best_score_),
        'training_only_imputation_and_feature_scaling': True,
        'targets_standardized_inside_each_training_fold': True}
    (OUT / 'settings.json').write_text(json.dumps(settings, indent=2))
    combined = pd.concat([pd.read_csv(RUN / 'validation_model_comparison.csv'), metrics], ignore_index=True)
    combined.to_csv(OUT / 'validation_comparison_all_models.csv', index=False)
    print(json.dumps(settings, indent=2))
    print(combined[['model','target','mae_mmHg','rmse_mmHg','bias_mmHg']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
