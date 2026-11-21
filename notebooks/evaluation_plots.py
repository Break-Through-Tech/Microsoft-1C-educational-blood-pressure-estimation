"""Evaluation plots and reports for a trained BP model (default: resnet_bp_v3).

Rebuilds the test set exactly as model_training.ipynb does (same preprocessing,
quality filter and recording split, seed 42), predicts with the saved model and
writes everything to notebooks/results/<model>/:

    summary.md                model, settings, training, test metrics and key takeaways
    metrics.csv               Target, Model, MAE, RMSE, R2, ME, SD, Slope, AAMI (model + mean baseline)
    metrics_by_bp_range.csv   Target, Range, Model, n, MAE, ME per SBP / DBP range
    scatter.png               predicted vs. actual, with the perfect-prediction diagonal
    bland_altman.png          error vs. mean BP, with bias and +/-1.96 SD limits
    error_by_range.png        MAE per SBP / DBP range

The dataset switches (input channels, ALIGN, STRICT_QC) are read from the checkpoint,
so the same script works for later models. No GPU needed: nothing is trained. Building
the dataset takes a while, so the test predictions are cached in test_predictions.npz
and reused on the next run (pass --rebuild to redo them).

Usage (from the project root or notebooks/):
    python notebooks/evaluation_plots.py
    python notebooks/evaluation_plots.py --model resnet_bp_v4a --data-dir /content/data
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

FS = 125
WINDOW_SIZE = 625
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:   # pasted into a notebook cell: assume the kernel runs in the project root or notebooks/
    HERE = os.path.abspath('notebooks' if os.path.isdir('notebooks') else '.')
ROOT = os.path.dirname(HERE)

# checkpoints saved before the experiment switches existed are the v1 baseline
BASELINE_CONFIG = {'use_gap': False, 'kernel': 3, 'input_channels': ['ppg'],
                   'loss': 'mse', 'weight_decay': 0.0, 'augment': False, 'conv_dropout': 0.0,
                   'align': False, 'strict_qc': False}


# ---------------------------------------------------------------------------
# Dataset: same steps as model_training.ipynb (keep the two in sync)
# ---------------------------------------------------------------------------

def remove_mean(x):
    import neurokit2 as nk
    return nk.signal_detrend(np.asarray(x, dtype=float), order=0)


def bandpass_filter(x, fs=FS, low=0.5, high=5.0, order=4):
    import neurokit2 as nk
    return nk.signal_filter(x, sampling_rate=fs, lowcut=low, highcut=high,
                            method='butterworth', order=order)


def normalize(x):
    peak = np.max(np.abs(x))
    return x / peak if peak > 0 else x


def get_bp_labels(abp_window, fs=FS, strict=False):
    from scipy.signal import find_peaks
    min_gap = fs * 0.4
    peaks, _ = find_peaks(abp_window, distance=min_gap)
    valleys, _ = find_peaks(-abp_window, distance=min_gap)
    if len(peaks) == 0 or len(valleys) == 0:
        return None
    sys_vals = abp_window[peaks]
    sys_vals = sys_vals[(sys_vals >= 70) & (sys_vals <= 220)]
    dia_vals = abp_window[valleys]
    dia_vals = dia_vals[(dia_vals >= 30) & (dia_vals <= 140)]
    if len(sys_vals) == 0 or len(dia_vals) == 0:
        return None
    if strict and len(peaks) >= 2 and not 35 <= 60 * fs / np.median(np.diff(peaks)) <= 140:
        return None
    sbp, dbp = np.median(sys_vals), np.median(dia_vals)
    if sbp <= dbp or sbp - dbp < 10:
        return None
    return sbp, dbp


def good_ppg_window(ppg_win, fs=FS, min_hr=48, max_hr=180):
    from scipy.signal import find_peaks
    if not np.isfinite(ppg_win).all() or np.std(ppg_win) < 1e-3:
        return False
    peaks, _ = find_peaks(ppg_win, distance=int(0.33 * fs), prominence=0.2)
    n_sec = len(ppg_win) / fs
    return min_hr * n_sec / 60 <= len(peaks) <= max_hr * n_sec / 60


def strict_ppg_window(ppg_win, fs=FS, min_hr=35, max_hr=140, max_ibi_cv=0.15, max_amp_sd=0.2):
    from scipy.signal import find_peaks
    from scipy.stats import skew
    peaks, _ = find_peaks(ppg_win, distance=int(0.33 * fs), prominence=0.2)
    valleys, _ = find_peaks(-ppg_win, distance=int(0.33 * fs), prominence=0.2)
    if len(peaks) < 3 or len(valleys) < 3:
        return False
    ibi = np.diff(peaks) / fs
    return (min_hr <= 60 / np.median(ibi) <= max_hr
            and ibi.std() / ibi.mean() <= max_ibi_cv
            and np.std(ppg_win[peaks]) <= max_amp_sd and np.std(ppg_win[valleys]) <= max_amp_sd
            and skew(ppg_win) >= 0)


def align_lag(ppg_filtered, abp, fs=FS, max_lag=FS):
    from scipy.signal import correlate, correlation_lags
    abp_filtered = bandpass_filter(remove_mean(abp))
    corr = correlate(ppg_filtered, abp_filtered, method='fft')
    lags = correlation_lags(len(ppg_filtered), len(abp_filtered))
    keep = np.abs(lags) <= max_lag
    return int(lags[keep][np.argmax(corr[keep])])


def build_dataset(data_dir, cfg):
    """X (N, channels, 625), y (N, 2) in mmHg, recording ids - as in the notebook."""
    import h5py
    channels = cfg['input_channels']
    need_deriv = 'd1' in channels or 'd2' in channels
    need_ecg = 'ecg' in channels
    x_dtype = np.float32 if len(channels) == 1 else np.float16

    X, y, rec_ids = [], [], []
    rec_counter = 0
    for part in range(1, 5):
        with h5py.File(os.path.join(data_dir, f'Part_{part}.mat'), 'r') as f:
            data = f[f'Part_{part}']
            for i in range(data.shape[0]):
                record = np.array(f[data[i, 0]]).T
                ppg, abp, ecg = record[0], record[1], record[2]
                if not (np.isfinite(ppg).all() and np.isfinite(abp).all()
                        and (not need_ecg or np.isfinite(ecg).all())):
                    rec_counter += 1
                    continue

                filtered = bandpass_filter(remove_mean(ppg))
                signals = {'ppg': filtered}
                if need_deriv:
                    signals['d1'] = np.gradient(filtered)
                    signals['d2'] = np.gradient(signals['d1'])
                if need_ecg:
                    signals['ecg'] = bandpass_filter(remove_mean(ecg), low=0.5, high=40.0)

                if cfg.get('align', False):
                    lag = align_lag(filtered, abp)
                    if lag > 0:
                        signals = {k: v[lag:] for k, v in signals.items()}
                        abp = abp[:-lag]
                    elif lag < 0:
                        signals = {k: v[:lag] for k, v in signals.items()}
                        abp = abp[-lag:]
                    filtered = signals['ppg']

                strict = cfg.get('strict_qc', False)
                for j in range(len(filtered) // WINDOW_SIZE):
                    s, e = j * WINDOW_SIZE, (j + 1) * WINDOW_SIZE
                    win = normalize(filtered[s:e])
                    labels = get_bp_labels(abp[s:e], strict=strict)
                    if labels is None or not good_ppg_window(win) or (strict and not strict_ppg_window(win)):
                        continue
                    X.append(np.stack([normalize(signals[c][s:e]) for c in channels]).astype(x_dtype))
                    y.append(labels)
                    rec_ids.append(rec_counter)
                rec_counter += 1
        print(f'Part_{part} done: {len(X):,} windows kept so far')
    return np.stack(X), np.array(y, dtype=np.float32), np.array(rec_ids)


def test_indices(rec_ids):
    """Test windows of the notebook's 70/15/15 split by recording (seed 42)."""
    rng = np.random.default_rng(42)
    unique_recs = rng.permutation(np.unique(rec_ids))
    n_train = int(0.70 * len(unique_recs))
    n_val = int(0.15 * len(unique_recs))
    return np.flatnonzero(np.isin(rec_ids, unique_recs[n_train + n_val:]))


# ---------------------------------------------------------------------------
# Model: same architecture as model_training.ipynb
# ---------------------------------------------------------------------------

def build_model(cfg):
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0):
            super().__init__()
            pad = kernel_size // 2
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.skip = (nn.Identity() if in_ch == out_ch else
                         nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch)))
            self.pool = nn.MaxPool1d(2)
            self.drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.relu(out + self.skip(x))
            return self.drop(self.pool(out))

    class PPGResNet(nn.Module):
        def __init__(self, input_len=WINDOW_SIZE, in_channels=1, filters=(32, 64, 128), kernel_size=3,
                     use_gap=False, conv_dropout=0.0, n_outputs=2):
            super().__init__()
            blocks, in_ch = [], in_channels
            for f in filters:
                blocks.append(ResidualBlock(in_ch, f, kernel_size, conv_dropout))
                in_ch = f
            self.features = nn.Sequential(*blocks)
            if use_gap:
                flatten = [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
                n_flat = in_ch
            else:
                flatten = [nn.Flatten()]
                n_flat = in_ch * (input_len // 2 ** len(filters))
            self.head = nn.Sequential(*flatten, nn.Linear(n_flat, 512), nn.ReLU(inplace=True),
                                      nn.Dropout(0.5), nn.Linear(512, n_outputs))

        def forward(self, x):
            return self.head(self.features(x))

    return PPGResNet(in_channels=len(cfg['input_channels']), kernel_size=cfg['kernel'],
                     use_gap=cfg['use_gap'], conv_dropout=cfg.get('conv_dropout', 0.0))


def test_predictions(model_path, data_dir):
    """(pred, true) in mmHg for every test window, shape (N, 2) each, and their recording ids."""
    import torch
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', BASELINE_CONFIG)
    print(f'Config: {cfg}')

    X, y, rec_ids = build_dataset(data_dir, cfg)
    idx = test_indices(rec_ids)
    print(f'{len(idx):,} test windows')

    model = build_model(cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            xb = torch.from_numpy(X[idx[s:s + 1024]]).to(device).float()
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds) * ckpt['y_std'] + ckpt['y_mean']
    return pred.astype(np.float32), y[idx], rec_ids[idx]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

TARGETS = ['SBP', 'DBP']
COLORS = {'SBP': '#2a78d6', 'DBP': '#eb6834'}       # categorical slots 1 and 2
INK, INK_2, INK_MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
# BP ranges per target, by that value alone: (label, lower bound inclusive, upper bound exclusive)
RANGES = {
    'SBP': [('Low (<90)', -np.inf, 90), ('Normal (90-119)', 90, 120), ('Elevated (120-129)', 120, 130),
            ('Stage 1 (130-139)', 130, 140), ('Stage 2 (≥140)', 140, np.inf)],
    'DBP': [('Low (<60)', -np.inf, 60), ('Normal (60-79)', 60, 80), ('Stage 1 (80-89)', 80, 90),
            ('Stage 2 (≥90)', 90, np.inf)],
}
BASELINE_LABEL = 'Mean baseline'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.frameon': False, 'legend.fontsize': 9, 'font.size': 10,
})


def model_label(model_name):
    """'resnet_bp_v3' -> '1D ResNet (v3)'; the first checkpoint 'resnet_bp' is v1."""
    return f"1D ResNet ({model_name.replace('resnet_bp_', '').replace('resnet_bp', 'v1')})"


def plot_scatter(pred, true, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for k, (ax, target) in enumerate(zip(axes, TARGETS)):
        t, p = true[:, k], pred[:, k]
        lo, hi = np.percentile(np.r_[t, p], [0.2, 99.8])
        lo, hi = lo - 5, hi + 5
        ax.scatter(t, p, s=3, alpha=0.06, color=COLORS[target], linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], '--', color=INK_2, lw=1.2, label='Perfect prediction')
        slope, intercept = np.polyfit(t, p, 1)
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color=INK, lw=2,
                label=f'Best fit (slope {slope:.2f})')
        mae = np.abs(p - t).mean()
        r = np.corrcoef(t, p)[0, 1]
        ax.text(0.03, 0.97, f'MAE {mae:.1f} mmHg\nr = {r:.2f}\nn = {len(t):,} windows',
                transform=ax.transAxes, va='top', color=INK, fontsize=9)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal',
               xlabel=f'Actual {target} (mmHg)', ylabel=f'Predicted {target} (mmHg)', title=target)
        ax.legend(loc='lower right')
    fig.suptitle(f'{title}: predicted vs. actual (test set)', color=INK, fontsize=13, fontweight='bold')
    fig.text(0.5, 0.01, 'A best-fit slope below 1 means predictions are squashed toward the average.',
             ha='center', color=INK_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_bland_altman(pred, true, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k, (ax, target) in enumerate(zip(axes, TARGETS)):
        mean = (pred[:, k] + true[:, k]) / 2
        diff = pred[:, k] - true[:, k]
        bias, sd = diff.mean(), diff.std()
        upper, lower = bias + 1.96 * sd, bias - 1.96 * sd
        ax.scatter(mean, diff, s=3, alpha=0.06, color=COLORS[target], linewidths=0, rasterized=True)
        ax.axhline(0, color=GRID, lw=1.2)
        ax.axhline(bias, color=INK, lw=2)
        for v in (upper, lower):
            ax.axhline(v, color=INK_2, lw=1.2, ls='--')
        xlo, xhi = np.percentile(mean, [0.2, 99.8])
        ylim = max(abs(upper), abs(lower)) * 1.6
        ax.set(xlim=(xlo, xhi), ylim=(-ylim, ylim), title=target,
               xlabel=f'Mean of predicted and actual {target} (mmHg)',
               ylabel=f'Predicted − actual {target} (mmHg)')
        for v, label in ((upper, '+1.96 SD'), (bias, 'Bias'), (lower, '−1.96 SD')):
            ax.annotate(f'{label}: {v:+.1f}', xy=(1, v), xycoords=('axes fraction', 'data'),
                        xytext=(-4, 4), textcoords='offset points', ha='right', va='bottom',
                        color=INK, fontsize=9)
    fig.suptitle(f'{title}: Bland-Altman (test set)', color=INK, fontsize=13, fontweight='bold')
    fig.text(0.5, 0.01, 'A downward slope means high BP is under-predicted and low BP over-predicted.',
             ha='center', color=INK_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_error_by_range(pred, true, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [5, 4]})
    for k, (ax, target) in enumerate(zip(axes, TARGETS)):
        err = np.abs(pred[:, k] - true[:, k])
        labels, maes = [], []
        for name, lo, hi in RANGES[target]:
            m = (true[:, k] >= lo) & (true[:, k] < hi)
            labels.append(f'{name.replace(" (", chr(10) + "(")}\nn = {m.sum():,}')
            maes.append(err[m].mean() if m.any() else np.nan)
        bars = ax.bar(np.arange(len(maes)), maes, 0.6, color=COLORS[target])
        ax.bar_label(bars, fmt='%.1f', padding=3, color=INK, fontsize=9)
        ax.set_xticks(np.arange(len(maes)), labels)
        ax.grid(axis='x', visible=False)
        ax.set(ylabel='MAE (mmHg)', xlabel=f'True {target} range (mmHg)', title=target)
        ax.margins(y=0.15)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax)        # same scale, so SBP and DBP bars compare directly
    fig.suptitle(f'{title}: error by BP range (test set)', color=INK, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def make_plots(pred, true, model_name, out_dir):
    title = model_name.replace('resnet_bp_', '').replace('resnet_bp', 'v1')
    paths = []
    for suffix, fn in (('scatter', plot_scatter), ('bland_altman', plot_bland_altman),
                       ('error_by_range', plot_error_by_range)):
        fig = fn(pred, true, title)
        path = os.path.join(out_dir, f'{suffix}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths




# ---------------------------------------------------------------------------
# Reports: metrics.csv, metrics_by_bp_range.csv, summary.md
# ---------------------------------------------------------------------------

AHA_CLASSES = ['Normal', 'Elevated', 'HTN stage 1', 'HTN stage 2']
# BHS protocol: share of errors within 5 / 10 / 15 mmHg needed for each grade
BHS_GRADES = [('A', (60, 85, 95)), ('B', (50, 75, 90)), ('C', (40, 65, 85))]
SETTING_NOTES = {
    'use_gap': 'global average pooling before the dense head',
    'kernel': 'conv kernel size (samples at 125 Hz)',
    'input_channels': 'input signals per window',
    'loss': 'training loss on standardized targets',
    'weight_decay': 'AdamW weight decay (0 = plain Adam)',
    'augment': 'random time shift + noise on training windows',
    'conv_dropout': 'channel dropout after each residual block',
    'align': 'ABP shifted onto the PPG before labelling',
    'strict_qc': 'stricter window quality filter (benchmark paper)',
}


def bhs_grade(within):
    for grade, needed in BHS_GRADES:
        if all(w >= n for w, n in zip(within, needed)):
            return grade
    return 'D'


def error_metrics(pred, true, baseline_mae=None):
    """One dict per target (SBP, DBP)."""
    rows = []
    for k, target in enumerate(TARGETS):
        err = pred[:, k] - true[:, k]
        abs_err = np.abs(err)
        within = [100 * (abs_err <= t).mean() for t in (5, 10, 15)]
        constant = np.std(pred[:, k]) < 1e-6          # the baseline: r and slope are undefined
        rows.append({
            'target': target, 'n_windows': len(err),
            'mae': abs_err.mean(), 'rmse': np.sqrt((err ** 2).mean()),
            'mase_pct': 100 * abs_err.mean() / baseline_mae[k] if baseline_mae is not None else 100.0,
            'bias_me': err.mean(), 'sd_error': err.std(),
            'loa_lower': err.mean() - 1.96 * err.std(), 'loa_upper': err.mean() + 1.96 * err.std(),
            'pearson_r': np.nan if constant else np.corrcoef(true[:, k], pred[:, k])[0, 1],
            'r2': 1 - (err ** 2).sum() / ((true[:, k] - true[:, k].mean()) ** 2).sum(),
            'fit_slope': 0.0 if constant else np.polyfit(true[:, k], pred[:, k], 1)[0],
            'within_5_pct': within[0], 'within_10_pct': within[1], 'within_15_pct': within[2],
            'bhs_grade': bhs_grade(within),
            'aami_pass': bool(abs(err.mean()) <= 5 and err.std() <= 8),
        })
    return rows


def range_metrics(pred, true, y_mean):
    """One dict per target and range: n, model MAE / ME, and the mean baseline's MAE / ME."""
    rows = []
    for k, target in enumerate(TARGETS):
        for name, lo, hi in RANGES[target]:
            m = (true[:, k] >= lo) & (true[:, k] < hi)
            err = pred[m, k] - true[m, k]
            base_err = y_mean[k] - true[m, k]
            rows.append({'target': target, 'range': name, 'n': int(m.sum()),
                         'mae': np.abs(err).mean() if m.any() else np.nan,
                         'me': err.mean() if m.any() else np.nan,
                         'base_mae': np.abs(base_err).mean() if m.any() else np.nan,
                         'base_me': base_err.mean() if m.any() else np.nan})
    return rows


def aha_scores(pred, true):
    from sklearn.metrics import accuracy_score, f1_score

    def aha(bp):
        sbp, dbp = bp[:, 0], bp[:, 1]
        return np.select([(sbp >= 140) | (dbp >= 90), (sbp >= 130) | (dbp >= 80), sbp >= 120], [3, 2, 1], 0)

    t, p = aha(true), aha(pred)
    return accuracy_score(t, p), f1_score(t, p, labels=range(4), average='macro', zero_division=0)


def training_info(ckpt):
    h = ckpt.get('history')
    if not h:
        return None
    val = (np.array(h['val_mae_sbp']) + np.array(h['val_mae_dbp'])) / 2
    b = int(val.argmin())
    info = {'epochs': len(val), 'best_epoch': b + 1,
            'val_sbp': h['val_mae_sbp'][b], 'val_dbp': h['val_mae_dbp'][b]}
    if 'train_mae_sbp' in h:
        info.update(train_sbp=h['train_mae_sbp'][b], train_dbp=h['train_mae_dbp'][b])
    return info


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in rows]
    return '\n'.join(lines)


def takeaways(model_rows, base_rows, ranges, aha_acc, aha_f1):
    s, d = model_rows
    margin = ('Clearly better than guessing the average' if max(s['mase_pct'], d['mase_pct']) < 80
              else 'Only modestly better than guessing the average')
    out = [
        f'**{margin}.** Test MAE is {s["mae"]:.1f} mmHg for SBP and {d["mae"]:.1f} for DBP, against '
        f'{base_rows[0]["mae"]:.1f} and {base_rows[1]["mae"]:.1f} for always predicting the training mean '
        f'(MASE {s["mase_pct"]:.0f}% and {d["mase_pct"]:.0f}%; 100% = no better than the mean).',
        f'**Predictions are squashed toward the average.** The best-fit slope of predicted on actual is '
        f'{s["fit_slope"]:.2f} for SBP and {d["fit_slope"]:.2f} for DBP (1.0 would be perfect), so the model '
        f'follows the trend but compresses it.',
    ]
    sbp = [r for r in ranges if r['target'] == 'SBP' and r['n'] > 0]
    lo, hi = sbp[0], sbp[-1]
    worst = max(sbp, key=lambda r: r['mae'])
    best = min(sbp, key=lambda r: r['mae'])
    text = (f'**Errors are largest at the extremes.** SBP MAE is highest in {worst["range"]} '
            f'({worst["mae"]:.1f} mmHg) and lowest in {best["range"]} ({best["mae"]:.1f}). '
            f'SBP {hi["range"]} readings are under-predicted on average (ME {hi["me"]:+.1f} mmHg) and '
            f'{lo["range"]} readings over-predicted ({lo["me"]:+.1f}), the pattern the squashing predicts.')
    beaten = [r for r in sbp if r['mae'] > r['base_mae']]
    if beaten:
        names = [r['range'] for r in beaten]
        names = ' and '.join([', '.join(names[:-1]), names[-1]] if len(names) > 1 else names)
        text += (f' The low errors in the middle are partly free: those readings sit near the average, and in '
                 f'{names} the mean predictor does even better on SBP '
                 f'({", ".join(f"{r["base_mae"]:.1f} vs. {r["mae"]:.1f}" for r in beaten)} mmHg). '
                 f'The model earns its gain at the extremes.')
    out.append(text)
    fails = [r['target'] for r in model_rows if not r['aami_pass']]
    aami = {0: 'both SBP and DBP pass', 1: f'{fails[0] if fails else ""} fails', 2: 'both SBP and DBP fail'}[len(fails)]
    out.append(
        f'**Not clinically accurate.** Against the AAMI standard (mean error within ±5 mmHg, SD at most 8), '
        f'{aami} (SD {s["sd_error"]:.1f} / {d["sd_error"]:.1f}). BHS grades: SBP {s["bhs_grade"]}, '
        f'DBP {d["bhs_grade"]} (A is best, D fails). Only {s["within_10_pct"]:.0f}% of SBP predictions '
        f'are within 10 mmHg. That is expected for an educational prototype, but it rules out clinical use.')
    out.append(
        f'**AHA category from predicted BP:** accuracy {aha_acc:.1%}, macro-F1 {aha_f1:.3f} '
        f'(4 classes). The Elevated band is only 10 mmHg wide, narrower than the typical SBP error.')
    out.append(
        '**Caveat on the split:** windows are split by recording, so no recording appears in both train and '
        'test. The UCI dataset has no patient IDs, though, and one patient can have several recordings, so '
        'these numbers may still be somewhat optimistic for truly unseen people.')
    return out


def write_reports(pred, true, rec_ids, ckpt, model_name, out_dir):
    import csv
    cfg = ckpt.get('config', BASELINE_CONFIG)
    y_mean = np.asarray(ckpt['y_mean'])

    base_rows = error_metrics(np.broadcast_to(y_mean, true.shape).astype(np.float64), true)
    baseline_mae = [r['mae'] for r in base_rows]
    model_rows = error_metrics(pred, true, baseline_mae)
    ranges = range_metrics(pred, true, y_mean)
    aha_acc, aha_f1 = aha_scores(pred, true)

    def write_csv(path, rows):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return path

    label = model_label(model_name)
    metrics_rows = []
    for m, b in zip(model_rows, base_rows):
        for name, r in ((label, m), (BASELINE_LABEL, b)):
            metrics_rows.append({'Target': r['target'], 'Model': name, 'MAE': round(r['mae'], 2),
                                 'RMSE': round(r['rmse'], 2), 'R2': round(r['r2'], 3) + 0.0, 'ME': round(r['bias_me'], 2),
                                 'SD': round(r['sd_error'], 2), 'Slope': round(r['fit_slope'], 3),
                                 'AAMI': 'Pass' if r['aami_pass'] else 'Fail'})
    range_rows = [{'Target': r['target'], 'Range': r['range'], 'Model': name, 'n': r['n'],
                   'MAE': round(r[mae], 2), 'ME': round(r[me], 2)}
                  for name, mae, me in ((label, 'mae', 'me'), (BASELINE_LABEL, 'base_mae', 'base_me'))
                  for r in ranges]
    paths = [write_csv(os.path.join(out_dir, 'metrics.csv'), metrics_rows),
             write_csv(os.path.join(out_dir, 'metrics_by_bp_range.csv'), range_rows)]

    n_params = sum(p.numel() for p in build_model(cfg).parameters())
    train = training_info(ckpt)
    md = [f'# {model_name}: evaluation summary', '',
          'Cuffless blood-pressure estimation from PPG: a 1D ResNet regressing systolic (SBP) and diastolic '
          '(DBP) pressure from one 5 s window (625 samples at 125 Hz). Generated by '
          '`notebooks/evaluation_plots.py`.', '',
          '## Model', '',
          f'- Checkpoint: `notebooks/resnet_models/{model_name}.pt`',
          f'- Architecture: 3 residual blocks (32, 64, 128 filters), '
          f'{"global average pooling" if cfg["use_gap"] else "flatten"} → dense 512 → dropout 0.5 → 2 outputs',
          f'- Parameters: {n_params:,}', '',
          '## Settings', '',
          md_table(['Setting', 'Value', 'Meaning'],
                   [[f'`{k}`', f'`{v}`', SETTING_NOTES.get(k, '')] for k, v in cfg.items()]), '']
    if train:
        md += ['## Training', '',
               f'- Ran {train["epochs"]} epochs; best validation epoch {train["best_epoch"]} (weights kept from that epoch)',
               f'- Validation MAE at that epoch: SBP {train["val_sbp"]:.2f}, DBP {train["val_dbp"]:.2f} mmHg']
        if 'train_sbp' in train:
            md.append(f'- Training MAE at that epoch: SBP {train["train_sbp"]:.2f}, DBP {train["train_dbp"]:.2f} mmHg')
        md.append('')
    n_recs = f' from {len(np.unique(rec_ids)):,} recordings' if rec_ids is not None else ''
    md += ['## Test results', '',
           f'{len(true):,} test windows{n_recs}, split by recording (70/15/15, seed 42).', '',
           md_table(['', 'MAE', 'RMSE', 'MASE', 'R²', 'ME ± SD', 'Slope', '≤5 / ≤10 / ≤15 mmHg', 'BHS', 'AAMI'],
                    [[f'**{r["target"]}**', f'{r["mae"]:.2f}', f'{r["rmse"]:.2f}', f'{r["mase_pct"]:.0f}%',
                      f'{r["r2"]:.3f}', f'{r["bias_me"]:+.2f} ± {r["sd_error"]:.2f}', f'{r["fit_slope"]:.2f}',
                      f'{r["within_5_pct"]:.0f}% / {r["within_10_pct"]:.0f}% / {r["within_15_pct"]:.0f}%',
                      r['bhs_grade'], 'pass' if r['aami_pass'] else 'fail'] for r in model_rows] +
                    [[f'Baseline {r["target"]}', f'{r["mae"]:.2f}', f'{r["rmse"]:.2f}', '100%',
                      f'{r["r2"]:.3f}', f'{r["bias_me"]:+.2f} ± {r["sd_error"]:.2f}', f'{r["fit_slope"]:.2f}',
                      f'{r["within_5_pct"]:.0f}% / {r["within_10_pct"]:.0f}% / {r["within_15_pct"]:.0f}%',
                      r['bhs_grade'], 'pass' if r['aami_pass'] else 'fail'] for r in base_rows]), '',
           'Errors in mmHg. Baseline: always predict the training-set mean. MASE: model MAE / baseline MAE. '
           'R²: share of BP variance explained. ME: mean error (bias). Slope: best-fit slope of predicted on actual (1 = no squashing). BHS: British Hypertension Society '
           'grade (A needs 60/85/95% within 5/10/15 mmHg). AAMI: |bias| ≤ 5 and SD ≤ 8 mmHg.', '',
           '### Error by BP range', '',
           md_table(['Target', 'Range', 'n', 'MAE', 'ME', 'Baseline MAE'],
                    [[r['target'], r['range'], f'{r["n"]:,}', f'{r["mae"]:.2f}', f'{r["me"]:+.2f}',
                      f'{r["base_mae"]:.2f}'] for r in ranges]), '',
           'Ranges use the true value of each target on its own. ME > 0: over-predicted on average.', '',
           '## Key takeaways', '']
    md += [f'{i}. {t}' for i, t in enumerate(takeaways(model_rows, base_rows, ranges, aha_acc, aha_f1), 1)]
    md += ['', '## Figures', '',
           '![Predicted vs. actual](scatter.png)', '',
           '![Bland-Altman](bland_altman.png)', '',
           '![Error by BP range](error_by_range.png)', '']
    path = os.path.join(out_dir, 'summary.md')
    with open(path, 'w') as f:
        f.write('\n'.join(md))
    return paths + [path]


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='resnet_bp_v3', help='checkpoint name in notebooks/resnet_models')
    # default: data/cuff_less_bp_estimation (the notebook's location), else the .mat files directly in data/
    default_data = os.path.join(ROOT, 'data', 'cuff_less_bp_estimation')
    if not os.path.exists(os.path.join(default_data, 'Part_1.mat')):
        default_data = os.path.join(ROOT, 'data')
    ap.add_argument('--data-dir', default=default_data, help='folder with Part_1.mat - Part_4.mat')
    ap.add_argument('--out-dir', default=None, help='default: notebooks/results/<model>')
    ap.add_argument('--rebuild', action='store_true', help='ignore cached predictions')
    # parse_known_args: Jupyter / the VS Code Interactive Window pass their own --f=kernel.json argument
    args, _ = ap.parse_known_args()

    out_dir = args.out_dir or os.path.join(HERE, 'results', args.model)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(HERE, 'resnet_models', f'{args.model}.pt')
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    cache = os.path.join(out_dir, 'test_predictions.npz')
    if os.path.exists(cache) and not args.rebuild:
        print(f'Using cached predictions from {cache}')
        saved = np.load(cache)
        pred, true = saved['pred'], saved['true']
        rec_ids = saved['rec_ids'] if 'rec_ids' in saved else None
    else:
        missing = [f'Part_{p}.mat' for p in range(1, 5)
                   if not os.path.exists(os.path.join(args.data_dir, f'Part_{p}.mat'))]
        if missing:
            raise SystemExit(f'Missing {missing} in {args.data_dir}. Download the UCI dataset there, '
                             f'pass --data-dir, or copy test_predictions.npz into {out_dir}.')
        pred, true, rec_ids = test_predictions(model_path, args.data_dir)
        np.savez(cache, pred=pred, true=true, rec_ids=rec_ids)
        print(f'Saved predictions to {cache}')

    for path in make_plots(pred, true, args.model, out_dir) + write_reports(pred, true, rec_ids, ckpt,
                                                                            args.model, out_dir):
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
