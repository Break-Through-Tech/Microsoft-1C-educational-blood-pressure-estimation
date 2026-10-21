# %% [markdown]
# ## SVR Model: SBP / DBP from PPG features
#
# Support Vector Regression on the hand-crafted PPG features from the EDA,
# compared against a naive (predict-the-mean) baseline.
#
# We work with **one Part_X.mat file at a time** to keep CPU / memory usage low.
# Features are cached to `data/features_part{PART}.csv`, so the slow NeuroKit
# step only runs once per part.

# %%
import os
import time
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import loguniform

from features import build_feature_table

warnings.filterwarnings('ignore')  # NeuroKit is noisy on short/noisy windows

SBP_COLOR = '#2a78d6'
DBP_COLOR = '#eb6834'

# %% [markdown]
# ### Settings
#
# - `PART` - which Part_X.mat file to use (1-4)
# - `WINDOWS_PER_RECORD` - max 5s windows per recording. Spreads the data across
#   patients instead of letting a few long recordings dominate, and keeps runtime down.
# - `N_JOBS` - CPU cores to use. 1 = gentlest on your laptop, -1 = all cores.
# - `TUNE_SAMPLE` - SVR training time grows ~quadratically with rows, so we tune
#   hyperparameters on a subsample, then refit the best model on the full train set.
# - `WEIGHT_POWER` - how strongly the weighted SVR up-weights rare BP values
#   (0 = no weighting, 0.5 = moderate, 1 = full inverse-frequency).

# %%
PART = 1
WINDOWS_PER_RECORD = 10
N_JOBS = 1
TUNE_SAMPLE = 4000
WEIGHT_POWER = 0.5
SEED = 42

# resolve paths from this file's location so the script runs from any folder
# (__file__ isn't defined in an interactive/Jupyter session, which already runs in notebooks/)
try:
    NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    NOTEBOOK_DIR = os.getcwd()
DATA_ROOT = os.path.join(NOTEBOOK_DIR, '..', 'data')
DATA_DIR = os.path.join(DATA_ROOT, 'cuff_less_bp_estimation')
CACHE_PATH = os.path.join(DATA_ROOT, f'features_part{PART}.csv')

# %% [markdown]
# ### Load (or build) the feature table

# %%
if os.path.exists(CACHE_PATH):
    df = pd.read_csv(CACHE_PATH)
    print(f'Loaded cached features from {CACHE_PATH}')
else:
    start = time.time()
    df = build_feature_table(
        f'{DATA_DIR}/Part_{PART}.mat', f'Part_{PART}',
        windows_per_record=WINDOWS_PER_RECORD, n_jobs=N_JOBS, seed=SEED,
    )
    df.to_csv(CACHE_PATH, index=False)
    print(f'Built features in {(time.time() - start) / 60:.1f} min -> {CACHE_PATH}')

print(f'{len(df)} windows from {df["record"].nunique()} recordings')
df.head()

# %% [markdown]
# ### Choose feature columns
#
# The fiducial features (dicrotic notch etc.) can only be found in a small share of
# windows on this dataset, so we drop any feature that is missing in more than half
# of the windows. The remaining gaps are filled with the median inside the pipeline.

# %%
label_cols = ['sbp', 'dbp']
id_cols = ['record', 'window']
candidate_cols = [c for c in df.columns if c not in label_cols + id_cols]

missing = df[candidate_cols].isna().mean()
feature_cols = missing[missing <= 0.5].index.tolist()
print(f'Using {len(feature_cols)} features; dropped (mostly missing): {missing[missing > 0.5].index.tolist()}')

df = df.replace([np.inf, -np.inf], np.nan)
X = df[feature_cols]
groups = df['record']

# %% [markdown]
# ### Train / test split by recording
#
# Windows from the same recording (same patient, minutes apart) look very similar.
# If they land in both train and test, the model is graded on near-copies of its
# training data and the scores look much better than they really are. So we split
# by **recording**: every window from a given recording is either all train or all test.

# %%
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
groups_train = groups.iloc[train_idx]
print(f'Train: {len(X_train)} windows ({groups_train.nunique()} recordings)')
print(f'Test:  {len(X_test)} windows ({groups.iloc[test_idx].nunique()} recordings)')

# %% [markdown]
# ### SVR pipeline + hyperparameter search
#
# - `SimpleImputer` - fills the occasional missing feature with the train median
# - `StandardScaler` - SVR is distance-based, so features must be on the same scale
# - `SVR(kernel='rbf')` - key hyperparameters:
#   - `C` - how hard the model tries to fit every point (higher = less regularization)
#   - `epsilon` - errors smaller than this (in mmHg) are ignored
#   - `gamma` - how far each training point's influence reaches
#
# We tune with `RandomizedSearchCV` using `GroupKFold`, so the cross-validation is
# also split by recording.

# %%
def make_svr():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('svr', SVR(kernel='rbf')),
    ])


param_dist = {
    'svr__C': loguniform(1e-1, 1e3),
    'svr__epsilon': [0.5, 1, 2, 5],
    'svr__gamma': ['scale', 0.001, 0.01, 0.05, 0.1],
}

rng = np.random.default_rng(SEED)
tune_idx = rng.choice(len(X_train), min(TUNE_SAMPLE, len(X_train)), replace=False)

models, best_params = {}, {}
for target in label_cols:
    y_train = df[target].iloc[train_idx]

    search = RandomizedSearchCV(
        make_svr(), param_dist, n_iter=20, cv=GroupKFold(n_splits=5),
        scoring='neg_mean_absolute_error', n_jobs=N_JOBS, random_state=SEED,
    )
    start = time.time()
    search.fit(X_train.iloc[tune_idx], y_train.iloc[tune_idx], groups=groups_train.iloc[tune_idx])
    print(f'{target.upper()}: best CV MAE {-search.best_score_:.2f} mmHg '
          f'({time.time() - start:.0f}s)  params={search.best_params_}')

    # refit best settings on the full training set
    model = make_svr().set_params(**search.best_params_)
    model.fit(X_train, y_train)
    models[target] = model
    best_params[target] = search.best_params_

# %% [markdown]
# ### Weighted SVR: give rare BP values more weight
#
# Most training windows have BP near the average, so the cheapest way for the model
# to keep its error low is to predict close to the average for everyone. That's the
# "regression to the mean" we see in the plots below.
#
# To push back, we give each training window a **sample weight** based on how rare its
# BP value is: windows in crowded BP ranges count less, windows in rare ranges
# (very low / very high BP) count more. Errors on rare cases then cost the model more.
#
# - BP is grouped into 10 mmHg bins; weight = (bin count) ^ -`WEIGHT_POWER`
# - weights are scaled to average 1 (so `C` keeps the same meaning) and capped at 10
#   so a handful of extreme windows can't dominate
#
# We reuse the hyperparameters tuned above rather than running the search again.
# Expect a trade-off: better at the extremes, slightly worse on average.

# %%
def rarity_weights(y, power=WEIGHT_POWER, bin_width=10, max_weight=10):
    bins = np.floor(np.asarray(y) / bin_width)
    _, inverse, counts = np.unique(bins, return_inverse=True, return_counts=True)
    w = counts[inverse].astype(float) ** -power
    w = w / w.mean()
    w = np.minimum(w, max_weight)
    return w / w.mean()


weighted_models = {}
for target in label_cols:
    y_train = df[target].iloc[train_idx]
    w = rarity_weights(y_train)
    start = time.time()
    model = make_svr().set_params(**best_params[target])
    model.fit(X_train, y_train, svr__sample_weight=w)
    weighted_models[target] = model
    print(f'{target.upper()}: weighted SVR fit in {time.time() - start:.0f}s '
          f'(weights range {w.min():.2f}-{w.max():.2f})')

# %% [markdown]
# ### Evaluate vs naive baseline
#
# The baseline always predicts the training-set mean. A useful model has to beat it.
#
# Besides MAE / RMSE / R², we report **ME ± SD** (mean error and standard deviation of
# the error). The AAMI standard for BP devices requires |ME| ≤ 5 mmHg and SD ≤ 8 mmHg.
# This is a prototype, not a medical device, but it's a helpful reference point.
#
# **Slope** is the slope of predicted vs actual BP: 1.0 = predictions cover the full
# range, 0 = the model always predicts the average. It measures the "squashing" directly.

# %%
def evaluate(y_true, y_pred):
    err = y_pred - y_true
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'ME': err.mean(),
        'SD': err.std(),
        'Slope': np.polyfit(y_true, y_pred, 1)[0],
        'AAMI': 'PASS' if abs(err.mean()) <= 5 and err.std() <= 8 else 'FAIL',
    }


rows, preds, weighted_preds = [], {}, {}
for target in label_cols:
    y_train = df[target].iloc[train_idx]
    y_test = df[target].iloc[test_idx]

    baseline = DummyRegressor(strategy='mean').fit(X_train, y_train)
    rows.append({'Target': target.upper(), 'Model': 'Baseline (mean)',
                 **evaluate(y_test, baseline.predict(X_test))})

    preds[target] = models[target].predict(X_test)
    rows.append({'Target': target.upper(), 'Model': 'SVR (RBF)',
                 **evaluate(y_test, preds[target])})

    weighted_preds[target] = weighted_models[target].predict(X_test)
    rows.append({'Target': target.upper(), 'Model': 'SVR (weighted)',
                 **evaluate(y_test, weighted_preds[target])})

results = pd.DataFrame(rows).set_index(['Target', 'Model']).round(2)
results.to_csv(os.path.join(DATA_ROOT, f'svr_results_part{PART}.csv'))
results

# %% [markdown]
# ### Predicted vs actual
#
# Points on the dashed line are perfect predictions. If points form a flat band
# around the mean, the model is mostly predicting the average.
# Top row: plain SVR. Bottom row: weighted SVR.

# %%
model_preds = {'SVR (RBF)': preds, 'SVR (weighted)': weighted_preds}

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='col')
for row, (model_name, p) in enumerate(model_preds.items()):
    for col, (target, color) in enumerate(zip(label_cols, [SBP_COLOR, DBP_COLOR])):
        ax = axes[row, col]
        y_test = df[target].iloc[test_idx]
        ax.scatter(y_test, p[target], s=8, alpha=0.4, color=color)
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, '--', color='#898781', linewidth=1)
        ax.set_xlabel(f'Actual {target.upper()} (mmHg)')
        ax.set_ylabel(f'Predicted {target.upper()} (mmHg)')
        r = results.loc[(target.upper(), model_name)]
        ax.set_title(f'{model_name} - {target.upper()} (MAE {r["MAE"]:.1f}, slope {r["Slope"]:.2f})')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bland-Altman plots
#
# The standard figure in BP-estimation papers. Each point is one window:
# - x-axis: average of actual and predicted BP
# - y-axis: prediction error (predicted - actual)
# - solid line: mean error (bias); dashed lines: 95% limits of agreement (bias ± 1.96 SD)
#
# A downward slope means the model overpredicts low BP and underpredicts high BP,
# i.e. it is regressing to the mean.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='col')
for row, (model_name, p) in enumerate(model_preds.items()):
    for col, (target, color) in enumerate(zip(label_cols, [SBP_COLOR, DBP_COLOR])):
        ax = axes[row, col]
        y_test = df[target].iloc[test_idx].values
        mean_bp = (y_test + p[target]) / 2
        diff = p[target] - y_test
        bias, sd = diff.mean(), diff.std()

        ax.scatter(mean_bp, diff, s=8, alpha=0.4, color=color)
        ax.axhline(bias, color='#3b3a36', linewidth=1.2, label=f'Bias {bias:+.1f}')
        for limit in (bias - 1.96 * sd, bias + 1.96 * sd):
            ax.axhline(limit, color='#898781', linestyle='--', linewidth=1,
                       label='95% limits' if limit < bias else None)

        # trend line makes the regression-to-the-mean slope easy to see
        slope, intercept = np.polyfit(mean_bp, diff, 1)
        xs = np.linspace(mean_bp.min(), mean_bp.max(), 50)
        ax.plot(xs, slope * xs + intercept, color='#e34948', linewidth=1.2, label=f'Trend (slope {slope:.2f})')

        ax.set_xlabel(f'Mean of actual & predicted {target.upper()} (mmHg)')
        ax.set_ylabel('Predicted - actual (mmHg)')
        ax.set_title(f'Bland-Altman - {model_name} - {target.upper()}\n'
                     f'limits {bias - 1.96 * sd:.1f} to {bias + 1.96 * sd:+.1f} mmHg')
        ax.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Error by BP range
#
# Overall MAE hides where the model fails. Here we bin the **actual** BP into
# clinical ranges (ACC/AHA categories, plus hypotension) and report error per bin.
# A positive mean error = overpredicting, negative = underpredicting.

# %%
bp_bins = {
    'sbp': ([0, 90, 120, 130, 140, 300],
            ['Low (<90)', 'Normal (90-119)', 'Elevated (120-129)', 'Stage 1 (130-139)', 'Stage 2 (≥140)']),
    'dbp': ([0, 60, 80, 90, 300],
            ['Low (<60)', 'Normal (60-79)', 'Stage 1 (80-89)', 'Stage 2 (≥90)']),
}

range_rows = []
for target in label_cols:
    edges, names = bp_bins[target]
    y_test = df[target].iloc[test_idx].values
    bins = pd.cut(y_test, edges, labels=names, right=False)
    for model_name, p in model_preds.items():
        err = p[target] - y_test
        for name in names:
            mask = np.asarray(bins == name)
            if mask.sum() == 0:
                continue
            range_rows.append({
                'Target': target.upper(), 'Range': name, 'Model': model_name, 'n': int(mask.sum()),
                'MAE': np.abs(err[mask]).mean(), 'ME': err[mask].mean(),
            })

range_results = pd.DataFrame(range_rows).round(2)
range_results.to_csv(os.path.join(DATA_ROOT, f'svr_results_by_range_part{PART}.csv'), index=False)

# side-by-side view: one row per range, MAE / ME for each model
range_wide = range_results.pivot_table(
    index=['Target', 'Range', 'n'], columns='Model', values=['MAE', 'ME'], sort=False,
)
print(range_wide.to_string())

# plain SVR in a light shade, weighted SVR in the full color
LIGHT = {SBP_COLOR: '#9fc3ee', DBP_COLOR: '#f5b89e'}
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, target, color in zip(axes, label_cols, [SBP_COLOR, DBP_COLOR]):
    names = bp_bins[target][1]
    x = np.arange(len(names))
    width = 0.38
    for offset, (model_name, shade) in zip((-width / 2, width / 2),
                                           [('SVR (RBF)', LIGHT[color]), ('SVR (weighted)', color)]):
        sub = (range_results[(range_results['Target'] == target.upper()) & (range_results['Model'] == model_name)]
               .set_index('Range').reindex(names))
        ax.bar(x + offset, sub['MAE'], width, color=shade, label=model_name)
    counts = (range_results[range_results['Target'] == target.upper()]
              .drop_duplicates('Range').set_index('Range').reindex(names)['n'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{name}\nn={n}' for name, n in zip(names, counts)], fontsize=8)
    ax.set_ylabel('MAE (mmHg)')
    ax.set_title(f'Error by actual {target.upper()} range')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature importance (permutation)
#
# SVR with an RBF kernel has no built-in feature importances, so we use
# **permutation importance**: shuffle one feature at a time on the test set and
# measure how much the MAE gets worse. Bigger increase = the model relies on it more.

# %%
top_n = 15
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, target, color in zip(axes, label_cols, [SBP_COLOR, DBP_COLOR]):
    y_test = df[target].iloc[test_idx]
    perm = permutation_importance(
        models[target], X_test, y_test, scoring='neg_mean_absolute_error',
        n_repeats=5, random_state=SEED, n_jobs=N_JOBS,
    )
    imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values().tail(top_n)
    ax.barh(imp.index, imp.values, color=color)
    ax.set_xlabel('Increase in MAE when shuffled (mmHg)')
    ax.set_title(f'Top {top_n} features - {target.upper()}')
plt.tight_layout()
plt.show()
