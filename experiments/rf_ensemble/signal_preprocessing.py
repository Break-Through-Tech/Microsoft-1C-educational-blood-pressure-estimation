"""Prepare a traceable PPG-only pilot; no model training or clinical validation."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

FS, WIDTH = 125, 625


def record(handle, rid):
    return handle[Path(handle.filename).stem][rid, 0]


def waveform(handle, rid):
    return np.asarray(handle[record(handle, rid)]).T


def trace_hash(x):
    return hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()


class ConnectedRecords:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, key):
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            previous = self.parent[key]
            self.parent[key] = root
            key = previous
        return root

    def union(self, left, right):
        left, right = sorted((self.find(left), self.find(right)))
        self.parent[right] = left


def key(part, rid):
    return f'{part}:{int(rid):04d}'


def anchors(x):
    # Every local maximum, without record-dependent prominence or distance.
    # Identical shifted subtraces can share anchors despite different offsets.
    for peak in find_peaks(x)[0]:
        start = int(peak) - FS // 2
        if start < 0 or start + FS > len(x):
            continue
        context = x[start:start + FS]
        if np.isfinite(context).all() and np.ptp(context) > 0:
            yield trace_hash(context), start


def overlap_groups(root, selected, output):
    inventory = pd.read_csv(root / 'outputs/exploration_2_review/all_parts_record_inventory.csv')
    graph = ConnectedRecords([key(r.part, r.record_id) for r in inventory.itertuples()])
    for _, rows in inventory.groupby('ppg_sha256'):
        if len(rows) > 1:
            keys = [key(r.part, r.record_id) for r in rows.itertuples()]
            for other in keys[1:]:
                graph.union(keys[0], other)
    exact_windows = pd.read_csv(root / 'outputs/exploration_2_review/part1_cross_record_exact_window_matches.csv')
    for _, rows in exact_windows.groupby('ppg_sha256'):
        ids = sorted(rows.record_id.unique())
        for other in ids[1:]:
            graph.union(key('Part_1', ids[0]), key('Part_1', other))
    # Anchor bank only for the pilot, scanned against every record in all parts.
    bank = {}
    with h5py.File(root / 'data/Part_1.mat', 'r') as handle:
        for rid in sorted(selected.record_id.unique()):
            for digest, offset in anchors(waveform(handle, rid)[0]):
                bank.setdefault(digest, {}).setdefault(key('Part_1', rid), offset)
    pairs = {}
    for path in sorted((root / 'data').glob('Part_*.mat')):
        print('Scanning shifted exact overlaps:', path.stem, flush=True)
        with h5py.File(path, 'r') as handle:
            for rid in range(len(handle[path.stem])):
                scanned = key(path.stem, rid)
                x = waveform(handle, rid)[0]
                for digest, offset in anchors(x):
                    for reference, ref_offset in bank.get(digest, {}).items():
                        if scanned == reference:
                            continue
                        graph.union(reference, scanned)
                        pair = tuple(sorted((reference, scanned)))
                        if pair not in pairs:
                            pairs[pair] = dict(pilot_record=reference, matched_record=scanned,
                                pilot_start_sample=ref_offset, matched_start_sample=offset,
                                matched_samples=FS, matching_anchor_count=0)
                        pairs[pair]['matching_anchor_count'] += 1
    matches = pd.DataFrame(pairs.values(), columns=['pilot_record', 'matched_record',
        'pilot_start_sample', 'matched_start_sample', 'matched_samples', 'matching_anchor_count'])
    matches.to_csv(output / 'pilot_shifted_exact_overlap_matches.csv', index=False)
    inventory['record_key'] = [key(r.part, r.record_id) for r in inventory.itertuples()]
    inventory['overlap_group'] = inventory.record_key.map(graph.find)
    inventory[['part', 'record_id', 'record_key', 'overlap_group']].to_csv(
        output / 'overlap_record_groups.csv', index=False)
    return graph, matches


def paired_labels(abp, prominence):
    """Full-record raw ABP peaks and troughs from complete peak-to-peak beats."""
    peaks, _ = find_peaks(abp, distance=round(.30 * FS), prominence=prominence)
    troughs = np.array([a + np.argmin(abp[a:b + 1]) for a, b in zip(peaks[:-1], peaks[1:])], dtype=int)
    return peaks, troughs


def window_labels(abp, peaks, troughs, start):
    mask = (peaks[:-1] >= start) & (peaks[1:] < start + WIDTH)
    left, right = peaks[:-1][mask], peaks[1:][mask]
    low = troughs[mask]
    if len(left) < 2:
        return dict(sbp=np.nan, dbp=np.nan, beats=len(left), reason='fewer_than_two_complete_beats')
    intervals = (right - left) / FS
    systolic, diastolic = abp[left], abp[low]
    # Existing pressure bounds are retained as provisional review rules.
    # Flag a window containing an implausible beat rather than drop that beat.
    valid = ((intervals >= .30) & (intervals <= 2.) & (systolic >= 70) &
             (systolic <= 220) & (diastolic >= 30) & (diastolic <= 140) &
             (systolic - diastolic >= 10))
    reason = '' if valid.all() else 'beat_outside_provisional_bounds'
    return dict(sbp=float(np.median(systolic)), dbp=float(np.median(diastolic)),
                beats=len(left), reason=reason)


def assign_splits(table, seed):
    groups = np.array(sorted(table.loc[table.eligible, 'overlap_group'].unique()))
    if len(groups) < 10:
        raise ValueError('Fewer than 10 eligible overlap groups; do not split this pilot.')
    np.random.default_rng(seed).shuffle(groups)
    nval = max(1, round(len(groups) * .15))
    ntest = max(1, round(len(groups) * .15))
    assignments = {g: 'validation' for g in groups[:nval]}
    assignments.update({g: 'test' for g in groups[nval:nval + ntest]})
    assignments.update({g: 'train' for g in groups[nval + ntest:]})
    table['split'] = table.overlap_group.map(assignments).fillna('excluded')
    table.loc[~table.eligible, 'split'] = 'excluded'
    assert table.loc[table.eligible].groupby('overlap_group').split.nunique().max() == 1
    return table


def prepare(root=Path('.'), output=None, seed=42):
    root = Path(root).resolve()
    output = Path(output) if output else root / 'outputs/resnet_setup'
    output.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(root / 'outputs/exploration_2_review/distributed_pilot_features.csv')
    graph, matches = overlap_groups(root, selected, output)
    rows, inputs = [], []
    with h5py.File(root / 'data/Part_1.mat', 'r') as handle:
        for rid, group in selected.groupby('record_id', sort=True):
            ppg, abp, _ = waveform(handle, rid)
            if not np.isfinite(ppg).all() or not np.isfinite(abp).all():
                raise ValueError(f'Record {rid} has nonfinite samples; preparation must be reviewed.')
            clean = nk.ppg_clean(ppg, sampling_rate=FS)
            detectors = {p: paired_labels(abp, p) for p in [5, 10, 15]}
            for source in group.sort_values('start_sample').itertuples():
                start = int(source.start_sample)
                window = clean[start:start + WIDTH]
                raw = ppg[start:start + WIDTH]
                measurements = {p: window_labels(abp, *detectors[p], start) for p in detectors}
                primary = measurements[10]
                medians = np.array([[m['sbp'], m['dbp']] for m in measurements.values()])
                all_labels = np.isfinite(medians).all()
                spread = np.ptp(medians, axis=0) if all_labels else [np.nan, np.nan]
                same_count = len({m['beats'] for m in measurements.values()}) == 1
                reasons = sorted({m['reason'] for m in measurements.values() if m['reason']})
                if not all_labels:
                    reasons.append('missing_detector_label')
                elif max(spread) > 5 or not same_count:
                    reasons.append('detector_disagreement')
                if start < FS or start + WIDTH > len(ppg) - FS:
                    reasons.append('full_record_filter_boundary')
                if not np.isfinite(window).all() or np.ptp(window) == 0:
                    reasons.append('invalid_ppg')
                if source.processing_status != 'ok' or source.n_ppg_peaks < 2:
                    reasons.append('ppg_processing_or_peaks_need_review')
                if primary['beats'] and abs(primary['beats'] + 1 - source.n_ppg_peaks) > 2:
                    reasons.append('ppg_abp_beat_count_disagreement')
                if np.isfinite(primary['sbp']) and np.isfinite(source.sbp):
                    if max(abs(primary['sbp'] - source.sbp), abs(primary['dbp'] - source.dbp)) > 5:
                        reasons.append('target_definition_change_gt5')
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    per_window = nk.ppg_clean(raw, sampling_rate=FS)
                if caught:
                    reasons.append('window_filter_warning')
                scale = max(float(np.std(window)), 1e-12)
                row = dict(part='Part_1', record_id=int(rid), window_id=int(source.window_id),
                    global_window=int(source.global_window), start_sample=start,
                    array_row=len(inputs), record_key=key('Part_1', rid),
                    overlap_group=graph.find(key('Part_1', rid)),
                    original_sbp=source.sbp, original_dbp=source.dbp,
                    sbp=primary['sbp'], dbp=primary['dbp'], complete_beats=primary['beats'],
                    sbp_prominence_spread=spread[0], dbp_prominence_spread=spread[1],
                    ppg_quality_mean=source.ppg_quality_mean,
                    filter_nrmse=float(np.sqrt(np.mean((window - per_window) ** 2)) / scale),
                    filter_correlation=float(np.corrcoef(window, per_window)[0, 1]),
                    raw_ppg_sha256=trace_hash(raw),
                    eligible=not reasons, review_reasons=' | '.join(sorted(set(reasons))),
                    target_status='provisional_consensus' if not reasons else 'needs_review')
                rows.append(row)
                inputs.append(window.astype('float32'))
    table = assign_splits(pd.DataFrame(rows), seed)
    eligible = table[table.eligible]
    assert eligible.groupby('raw_ppg_sha256').split.nunique().max() == 1
    for split in ['train', 'validation', 'test']:
        if len(eligible[eligible.split.eq(split)]) < 2:
            raise ValueError(f'Too few eligible windows in {split}')
    # Arrays contain PPG only. Metadata, ABP, quality scores and IDs are not inputs.
    np.savez_compressed(output / 'pilot_waveforms.npz',
        x=np.stack(inputs)[:, None, :], y=table[['sbp', 'dbp']].to_numpy(dtype='float32'))
    table.to_csv(output / 'pilot_manifest.csv', index=False)
    queue = table.loc[~table.eligible].copy()
    queue['manual_decision'] = ''
    queue['manual_notes'] = ''
    queue.to_csv(output / 'generated_review_queue.csv', index=False)
    summary = dict(seed=seed, input='full-record cleaned PPG only', sampling_rate=FS,
        window_samples=WIDTH, target_order=['sbp', 'dbp'], target_status='provisional, not clinically validated',
        preprocessing='NeuroKit ppg_clean default (Elgendi), applied to full records; 1s record-edge review margin',
        overlap_scope='full-record hashes all parts; exact window hashes Part 1; 1s peak-anchor exact matches of 60 pilot records against all 12000 records',
        overlap_limit='near-duplicates and overlap chains with no pilot anchor are not exhaustively detected; patient identity unavailable',
        matched_record_pairs=len(matches), selected_windows=len(table),
        eligible_windows=len(eligible), eligible_records=int(eligible.record_id.nunique()),
        eligible_groups=int(eligible.overlap_group.nunique()),
        split_counts=eligible.split.value_counts().to_dict(),
        review_counts=table.loc[~table.eligible].review_reasons.value_counts().to_dict(),
        median_filter_nrmse=float(table.filter_nrmse.median()),
        median_filter_correlation=float(table.filter_correlation.median()),
        numpy=np.__version__, neurokit2=nk.__version__)
    (output / 'preparation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.root, args.output, args.seed)
