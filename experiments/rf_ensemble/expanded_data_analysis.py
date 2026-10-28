"""Random recording-level sampling across all Parts and coach-feedback checks.

Generated results are provisional research outputs, not patient-level validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import warnings
import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from prepare_resnet_data import FS, WIDTH, waveform, paired_labels, window_labels, anchors, ConnectedRecords, key, trace_hash

DEFAULT_OUTPUT = 'outputs/expanded_eda_oct14'


def flat_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.where(edges == -1)[0] - np.where(edges == 1)[0], initial=0))


def rate_from_peaks(peaks):
    return float(60 / np.median(np.diff(peaks) / FS)) if len(peaks) >= 2 else np.nan


def choose_windows(root, recordings_per_part=200, windows_per_record=12, seed=20261014):
    rows, inventory = [], []
    for number in range(1, 5):
        part = f'Part_{number}'
        rng = np.random.default_rng(seed + number)
        with h5py.File(Path(root) / 'data' / f'{part}.mat', 'r') as handle:
            refs = handle[part]
            lengths = np.array([handle[r].shape[0] for r in refs[:, 0]])
            for rid, length in enumerate(lengths):
                inventory.append(dict(part=part, record_id=rid, record_key=key(part, rid),
                    samples=int(length), seconds=length / FS, complete_windows=int(length // WIDTH)))
            pool = np.flatnonzero(lengths >= WIDTH)
            chosen = np.sort(rng.choice(pool, min(recordings_per_part, len(pool)), replace=False))
            for rid in chosen:
                ids = np.sort(rng.choice(lengths[rid] // WIDTH,
                    min(windows_per_record, lengths[rid] // WIDTH), replace=False))
                for wid in ids:
                    rows.append(dict(part=part, record_id=int(rid), record_key=key(part, rid),
                        window_id=int(wid), start_sample=int(wid * WIDTH), analysis_pool='random_all_parts'))
    return pd.DataFrame(rows), pd.DataFrame(inventory)


def legacy_selection(inventory):
    rows, remaining = [], 500
    for s in inventory[inventory.part.eq('Part_1')].sort_values('record_id').itertuples():
        count = min(remaining, s.complete_windows)
        for wid in range(count):
            rows.append(dict(part=s.part, record_id=s.record_id, record_key=s.record_key,
                window_id=wid, start_sample=wid * WIDTH, analysis_pool='legacy_first500'))
        remaining -= count
        if remaining == 0:
            break
    return pd.DataFrame(rows)


def describe_signal(x, prefix):
    return {prefix + 'mean': float(np.mean(x)), prefix + 'std': float(np.std(x)),
        prefix + 'iqr': float(stats.iqr(x)), prefix + 'skew': float(stats.skew(x)),
        prefix + 'kurtosis': float(stats.kurtosis(x)),
        prefix + 'p05': float(np.quantile(x, .05)), prefix + 'p95': float(np.quantile(x, .95))}


def extract_features(raw, clean, peaks):
    feats = describe_signal(raw, 'ppg_raw_')
    feats.update(describe_signal(clean, 'ppg_clean_'))
    vpg = np.gradient(clean, 1 / FS)
    apg = np.gradient(vpg, 1 / FS)
    feats.update(describe_signal(vpg, 'vpg_'))
    feats.update(describe_signal(apg, 'apg_'))
    gaps = np.diff(peaks) / FS
    feats.update(ppg_hr_bpm=rate_from_peaks(peaks),
        ppg_interval_std=float(np.std(gaps)) if len(gaps) else np.nan,
        ppg_interval_cv=float(np.std(gaps) / np.mean(gaps)) if len(gaps) else np.nan)
    amplitudes, widths, areas = [], [], []
    for left, right in zip(peaks[:-1], peaks[1:]):
        pulse = clean[left:right]
        low = pulse.min()
        amplitude = clean[left] - low
        amplitudes.append(amplitude)
        widths.append(float((pulse > low + amplitude / 2).sum() / FS))
        areas.append(float(np.trapezoid(pulse - low, dx=1 / FS)))
    feats.update(ppg_amplitude_mean=float(np.mean(amplitudes)) if amplitudes else np.nan,
        ppg_amplitude_std=float(np.std(amplitudes)) if amplitudes else np.nan,
        ppg_half_height_duration=float(np.mean(widths)) if widths else np.nan,
        ppg_baseline_subtracted_area=float(np.mean(areas)) if areas else np.nan)
    frequency, power = welch(clean, fs=FS, nperseg=256)
    band = (frequency >= .5) & (frequency <= 5)
    band_power = power[band].sum()
    feats['ppg_dominant_frequency_hz'] = float(frequency[band][np.argmax(power[band])])
    feats['ppg_high_frequency_fraction'] = float(power[(frequency >= 2) & (frequency <= 5)].sum() / band_power) if band_power > 0 else np.nan
    return feats


def quality_reasons(row, quality_threshold=.8, minimum_ppg_peaks=3, minimum_complete_beats=3):
    reasons = []
    if row.get('processing_status') != 'ok':
        reasons.append('processing_failure_or_warning')
    if not np.isfinite(row.get('ppg_quality_mean', np.nan)) or row['ppg_quality_mean'] < quality_threshold:
        reasons.append('ppg_quality_below_threshold')
    if row.get('ppg_peak_count', 0) < minimum_ppg_peaks:
        reasons.append('too_few_ppg_peaks')
    if row.get('abp_complete_beats', 0) < minimum_complete_beats:
        reasons.append('too_few_complete_abp_beats')
    if row.get('beat_count_mismatch', True):
        reasons.append('ppg_abp_peak_count_mismatch')
    if row.get('rate_mismatch', True):
        reasons.append('ppg_abp_rate_mismatch')
    if row.get('abp_detector_disagreement', True):
        reasons.append('abp_detector_disagreement')
    if row.get('label_reason', ''):
        reasons.append('abp_label_needs_review')
    if not np.isfinite(row.get('sbp', np.nan)) or not np.isfinite(row.get('dbp', np.nan)):
        reasons.append('missing_target')
    if row.get('nonfinite_samples', True):
        reasons.append('nonfinite_samples')
    if row.get('constant_signal', True):
        reasons.append('constant_signal')
    if row.get('max_flat_extreme_samples', 999) >= 5:
        reasons.append('flat_signal_extreme_ge5')
    if row.get('filter_boundary', True):
        reasons.append('record_filter_boundary')
    return reasons


def process_windows(root, selection, inventory, output, quality_threshold=.8, min_ppg=3, min_abp=3):
    jobs = pd.concat([selection, legacy_selection(inventory)], ignore_index=True)
    rows, logs = [], []
    feature_names = None
    total = jobs.record_key.nunique()
    started = time.monotonic()
    for index, ((part, rid), windows) in enumerate(jobs.groupby(['part', 'record_id'], sort=True), 1):
        with h5py.File(Path(root) / 'data' / f'{part}.mat', 'r') as handle:
            ppg, abp, _ = waveform(handle, rid)
        status, error, warning_text = 'ok', '', ''
        clean, quality, pp = np.full(len(ppg), np.nan), np.full(len(ppg), np.nan), np.array([], dtype=int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                if not np.isfinite(ppg).all() or not np.isfinite(abp).all():
                    raise ValueError('Nonfinite stored PPG or ABP samples in the record')
                sig, info = nk.ppg_process(ppg, sampling_rate=FS)
                clean, quality = sig.PPG_Clean.to_numpy(), sig.PPG_Quality.to_numpy()
                pp = np.array(info['PPG_Peaks'], dtype=int)
            except Exception as exception:
                status, error = 'failed', f'{type(exception).__name__}: {exception}'
            warning_text = ' | '.join(sorted({str(w.message) for w in caught}))
        if warning_text and status == 'ok':
            status = 'warning'
        logs.append(dict(part=part, record_id=rid, record_key=key(part, rid),
            status=status, error=error, warnings=warning_text, selected_windows=len(windows)))
        detectors = {p: paired_labels(abp, p) for p in [5, 10, 15]} if np.isfinite(abp).all() else {}
        for s in windows.itertuples():
            start, stop = s.start_sample, s.start_sample + WIDTH
            raw, pressure, filtered = ppg[start:stop], abp[start:stop], clean[start:stop]
            ppg_peaks = pp[(pp >= start) & (pp < stop)]
            primary_peaks = detectors[10][0] if detectors else np.array([], dtype=int)
            abp_peaks = primary_peaks[(primary_peaks >= start) & (primary_peaks < stop)]
            estimates = {p: window_labels(abp, *value, start) for p, value in detectors.items()}
            primary = estimates.get(10, dict(sbp=np.nan, dbp=np.nan, beats=0, reason='missing_detector'))
            medians = np.array([[m['sbp'], m['dbp']] for m in estimates.values()])
            spread = np.ptp(medians, axis=0) if len(medians) == 3 and np.isfinite(medians).all() else [np.nan, np.nan]
            stable = len(estimates) == 3 and np.isfinite(spread).all() and max(spread) <= 5 and len({m['beats'] for m in estimates.values()}) == 1
            ppg_rate, abp_rate = rate_from_peaks(ppg_peaks), rate_from_peaks(abp_peaks)
            ratio = ppg_rate / abp_rate if np.isfinite(abp_rate) and abp_rate > 0 else np.nan
            q = quality[start:stop]
            flat = max(flat_run(v == extreme) for v in [raw, pressure]
                       for extreme in [v.min(), v.max()])
            row = dict(part=part, record_id=int(rid), record_key=s.record_key,
                window_id=int(s.window_id), start_sample=int(start), analysis_pool=s.analysis_pool,
                window_key=f'{s.record_key}:{s.window_id:04d}',
                processing_status=status, ppg_quality_mean=float(q.mean()) if np.isfinite(q).all() else np.nan,
                ppg_peak_count=len(ppg_peaks), abp_peak_count=len(abp_peaks),
                abp_complete_beats=primary['beats'], ppg_count_bpm=len(ppg_peaks) * 60 / 5,
                abp_count_bpm=len(abp_peaks) * 60 / 5, ppg_rate_bpm=ppg_rate,
                abp_rate_bpm=abp_rate, ppg_abp_rate_ratio=ratio,
                beat_count_mismatch=abs(len(ppg_peaks) - len(abp_peaks)) > 1,
                rate_mismatch=not np.isfinite(ratio) or not .75 <= ratio <= 1.25,
                possible_ppg_double_count=bool(np.isfinite(ratio) and 1.7 <= ratio <= 2.3),
                possible_abp_double_count=bool(np.isfinite(ratio) and 1 / 2.3 <= ratio <= 1 / 1.7),
                sbp=primary['sbp'], dbp=primary['dbp'],
                sbp_prominence_spread=spread[0], dbp_prominence_spread=spread[1],
                abp_detector_disagreement=not stable,
                label_reason=' | '.join(sorted({m['reason'] for m in estimates.values() if m['reason']})) if estimates else 'missing_detector',
                nonfinite_samples=not np.isfinite(raw).all() or not np.isfinite(pressure).all(),
                constant_signal=bool(np.ptp(raw) == 0 or np.ptp(pressure) == 0),
                max_flat_extreme_samples=flat,
                filter_boundary=start < FS or stop > len(ppg) - FS,
                raw_ppg_sha256=trace_hash(raw))
            for prominence, detector in detectors.items():
                row[f'abp_peak_count_p{prominence}'] = int(((detector[0] >= start) & (detector[0] < stop)).sum())
            if np.isfinite(filtered).all() and np.isfinite(raw).all() and len(ppg_peaks) >= 2:
                with warnings.catch_warnings(record=True) as feature_warnings:
                    warnings.simplefilter('always')
                    try:
                        features = extract_features(raw, filtered, ppg_peaks - start)
                        row.update(features)
                        feature_names = list(features)
                    except Exception as exception:
                        row['processing_status'] = 'feature_failed'
                        row['feature_error'] = str(exception)
                if feature_warnings:
                    row['feature_warnings'] = ' | '.join(sorted({str(w.message) for w in feature_warnings}))
                    row['processing_status'] = 'feature_warning'
            reasons = quality_reasons(row, quality_threshold, min_ppg, min_abp)
            row.update(eligible=not reasons, review_reasons=' | '.join(reasons),
                target_status='provisional_consensus' if not reasons else 'needs_review')
            rows.append(row)
        if index % 25 == 0 or index == total:
            print(f'Processed {index}/{total} records, {len(rows)} windows, {time.monotonic()-started:.0f}s', flush=True)
            pd.DataFrame(logs).to_csv(output / 'record_processing_log.csv', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'all_sampled_window_audit.csv', index=False)
    pd.DataFrame(logs).to_csv(output / 'record_processing_log.csv', index=False)
    (output / 'feature_columns.json').write_text(json.dumps(feature_names, indent=2))
    return table, feature_names


def group_overlaps(root, selection, inventory, output):
    """Selected record anchors scanned against all records; connect exact matches."""
    graph = ConnectedRecords(inventory.record_key.tolist())
    selected = set(selection.record_key)
    bank = {}
    print('Building exact-overlap anchor bank for sampled recordings', flush=True)
    for part, records in selection[['part', 'record_id']].drop_duplicates().groupby('part'):
        with h5py.File(Path(root) / 'data' / f'{part}.mat', 'r') as handle:
            for rid in records.record_id:
                for digest, offset in anchors(waveform(handle, rid)[0]):
                    bank.setdefault(digest, {}).setdefault(key(part, rid), offset)
    full_hashes, matches, seen_pairs, sources = {}, [], set(), []
    for part, records in inventory.groupby('part'):
        print('Scanning all records for exact/shared shifted PPG:', part, flush=True)
        path = Path(root) / 'data' / f'{part}.mat'
        sources.append(dict(file=path.name, bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
        with h5py.File(path, 'r') as handle:
            for s in records.itertuples():
                x = waveform(handle, s.record_id)[0]
                digest = trace_hash(x)
                if digest in full_hashes:
                    graph.union(full_hashes[digest], s.record_key)
                else:
                    full_hashes[digest] = s.record_key
                for digest, offset in anchors(x):
                    references = bank.get(digest)
                    if not references:
                        continue
                    for reference, ref_offset in references.items():
                        if reference == s.record_key:
                            continue
                        graph.union(reference, s.record_key)
                        pair = tuple(sorted([reference, s.record_key]))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            matches.append(dict(reference_record=reference, matched_record=s.record_key,
                                reference_start_sample=ref_offset, matched_start_sample=offset, context_samples=FS))
    inventory['overlap_group'] = inventory.record_key.map(graph.find)
    inventory.to_csv(output / 'all_parts_inventory_and_groups.csv', index=False)
    pd.DataFrame(matches, columns=['reference_record', 'matched_record', 'reference_start_sample',
        'matched_start_sample', 'context_samples']).to_csv(output / 'exact_overlap_matches.csv', index=False)
    (output / 'raw_source_files.json').write_text(json.dumps(sources, indent=2))
    return graph


def grouped_split(table, seed):
    """Choose group partitions balanced on Part/record counts, never targets."""
    eligible = table[table.eligible].copy()
    groups = np.array(sorted(eligible.overlap_group.unique()))
    if len(groups) < 20:
        raise ValueError('Too few overlap groups for a useful expanded split')
    records = eligible[['record_key', 'part', 'overlap_group']].drop_duplicates()
    rng = np.random.default_rng(seed)
    nv, nt = max(1, round(.15 * len(groups))), max(1, round(.15 * len(groups)))
    best, best_score = None, np.inf
    targets = {'train': .70, 'validation': .15, 'test': .15}
    for _ in range(100):
        shuffled = rng.permutation(groups)
        assignment = {g: 'validation' for g in shuffled[:nv]}
        assignment.update({g: 'test' for g in shuffled[nv:nv+nt]})
        assignment.update({g: 'train' for g in shuffled[nv+nt:]})
        rec_split = records.overlap_group.map(assignment)
        score = 0.
        for part in sorted(records.part.unique()):
            mask = records.part.eq(part)
            for split, fraction in targets.items():
                observed = float(rec_split[mask].eq(split).mean())
                score += (observed - fraction) ** 2
                if observed == 0:
                    score += 10
        if score < best_score:
            best, best_score = assignment, score
    table['split'] = table.overlap_group.map(best).fillna('excluded')
    table.loc[~table.eligible, 'split'] = 'excluded'
    for identity in ['record_key', 'overlap_group', 'raw_ppg_sha256']:
        assert table.loc[table.eligible].groupby(identity).split.nunique().max() == 1
    for part in table.part.unique():
        for split in targets:
            assert ((table.part == part) & (table.split == split)).any(), (part, split)
    return table, best_score


def grouped_correlations(table, features):
    # Keep the reserved test set out of feature ranking and correlation analysis.
    development = table[table.split.isin(['train', 'validation'])].copy()
    values = development[features + ['sbp', 'dbp']].replace([np.inf, -np.inf], np.nan)
    medians = development.groupby('record_key')[features + ['sbp', 'dbp']].median()
    centered = values - development.groupby('record_key')[features + ['sbp', 'dbp']].transform('mean')
    rows = []
    for feature in features:
        for target in ['sbp', 'dbp']:
            rows.append(dict(feature=feature, target=target,
                pooled_spearman=values[feature].corr(values[target], method='spearman'),
                record_median_spearman=medians[feature].corr(medians[target], method='spearman'),
                within_record_spearman=centered[feature].corr(centered[target], method='spearman'),
                complete_window_pairs=int(values[[feature,target]].notna().all(axis=1).sum()),
                complete_record_pairs=int(medians[[feature,target]].notna().all(axis=1).sum())))
    return pd.DataFrame(rows)


def prepare(root=Path('.'), output=None, recordings=200, windows=12, seed=20261014, quality=.8, min_ppg=3, min_abp=3):
    root = Path(root).resolve()
    output = Path(output) if output else root / DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    selected, inventory = choose_windows(root, recordings, windows, seed)
    selected.to_csv(output / 'random_window_selection.csv', index=False)
    inventory.to_csv(output / 'all_parts_record_inventory.csv', index=False)
    config = dict(seed=seed, recordings_per_part=recordings, max_windows_per_record=windows,
        sampling_rate=FS, window_samples=WIDTH, quality_method='NeuroKit templatematch on full-record cleaned PPG',
        ppg_quality_min=quality, minimum_ppg_peak_count=min_ppg, minimum_complete_abp_beats=min_abp,
        maximum_ppg_abp_peak_count_difference=1, allowed_ppg_abp_rate_ratio=[.75,1.25],
        abp_prominence_mmHg=[5,10,15], primary_prominence_mmHg=10,
        abp_minimum_peak_spacing_seconds=.30, max_target_spread_mmHg=5,
        flat_extreme_review_samples=5, filter_record_margin_seconds=1,
        protocol='Approximately 70/15/15 overlap groups, balanced on Part/record counts only; grouped correlations on development; test not scored',
        quality_scope='Provisional screening thresholds, not probabilities of correctness or clinical standards',
        overlap_limit='Exact full traces and peak-centered 1s subtraces; near-duplicates and chains with no selected anchor remain unresolved; IDs are recordings, not confirmed people',
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, neurokit2=nk.__version__)
    (output / 'settings.json').write_text(json.dumps(config, indent=2))
    table, features = process_windows(root, selected, inventory, output, quality, min_ppg, min_abp)
    graph = group_overlaps(root, selected, inventory, output)
    expanded = table[table.analysis_pool.eq('random_all_parts')].copy()
    expanded['overlap_group'] = expanded.record_key.map(graph.find)
    # Repeated identical selected windows add weight, not independent evidence.
    duplicate = expanded.duplicated('raw_ppg_sha256', keep='first')
    expanded['exact_duplicate_selected_window'] = duplicate
    expanded.loc[duplicate, 'eligible'] = False
    expanded.loc[duplicate, 'target_status'] = 'needs_review'
    expanded.loc[duplicate, 'review_reasons'] = expanded.loc[duplicate, 'review_reasons'].fillna('').apply(
        lambda text:(text + ' | ' if text else '') + 'duplicate_selected_window')
    expanded, balance = grouped_split(expanded, seed)
    expanded.to_csv(output / 'expanded_features_and_split.csv', index=False)
    queue = expanded[~expanded.eligible].copy()
    queue['manual_decision'], queue['manual_notes'] = '', ''
    queue.to_csv(output / 'generated_review_queue.csv', index=False)
    grouped_correlations(expanded, features).to_csv(output / 'development_grouped_correlations.csv', index=False)
    medians = expanded[expanded.split.isin(['train','validation'])].groupby(['part','record_key'])[features + ['sbp','dbp']].median()
    medians.to_csv(output / 'development_record_medians.csv')
    sensitivity = []
    for threshold in [.6,.7,.8,.9]:
        passing = expanded.apply(lambda row:not quality_reasons(row,threshold,min_ppg,min_abp) and not row.exact_duplicate_selected_window,axis=1)
        for part, part_rows in expanded.groupby('part'):
            kept = part_rows.loc[passing.loc[part_rows.index]]
            sensitivity.append(dict(part=part, quality_threshold=threshold, selected_windows=len(part_rows),
                retained_windows=len(kept), retained_records=kept.record_key.nunique()))
    pd.DataFrame(sensitivity).to_csv(output / 'quality_threshold_sensitivity.csv', index=False)
    summary = expanded.groupby('part').agg(selected_records=('record_key','nunique'),
        selected_windows=('window_key','size'), eligible_windows=('eligible','sum'),
        possible_ppg_double_count=('possible_ppg_double_count','sum'), possible_abp_double_count=('possible_abp_double_count','sum'))
    summary.to_csv(output / 'part_summary.csv')
    split = expanded[expanded.eligible].groupby(['part','split']).agg(windows=('window_key','size'),
        records=('record_key','nunique'), overlap_groups=('overlap_group','nunique'))
    split.to_csv(output / 'split_counts.csv')
    (output / 'preparation_summary.json').write_text(json.dumps(dict(selected_records=int(expanded.record_key.nunique()),
        selected_windows=len(expanded), eligible_windows=int(expanded.eligible.sum()),
        eligible_records=int(expanded.loc[expanded.eligible,'record_key'].nunique()),
        eligible_overlap_groups=int(expanded.loc[expanded.eligible,'overlap_group'].nunique()),
        split_counts=expanded.loc[expanded.eligible,'split'].value_counts().to_dict(),
        split_part_record_balance_score=balance, feature_count=len(features), test_evaluated=False),indent=2))
    print(summary.to_string(), flush=True)
    print(split.to_string(), flush=True)
    return expanded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--recordings-per-part', type=int, default=200)
    parser.add_argument('--windows-per-record', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    if args.recordings_per_part < 1 or args.windows_per_record < 1:
        parser.error('Sample sizes must be positive')
    prepare(args.root,args.output,args.recordings_per_part,args.windows_per_record,args.seed)
