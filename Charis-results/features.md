# Shared features — Charis

Both submitted models use the same 68 inputs: 37 PPG-derived and 31 ECG/cross-signal features. ABP provides reference labels only. Recording/window identifiers and target values are not inputs.

## PPG feature names

- `ppg_raw_mean`
- `ppg_raw_std`
- `ppg_raw_iqr`
- `ppg_raw_skew`
- `ppg_raw_kurtosis`
- `ppg_raw_p05`
- `ppg_raw_p95`
- `ppg_clean_mean`
- `ppg_clean_std`
- `ppg_clean_iqr`
- `ppg_clean_skew`
- `ppg_clean_kurtosis`
- `ppg_clean_p05`
- `ppg_clean_p95`
- `vpg_mean`
- `vpg_std`
- `vpg_iqr`
- `vpg_skew`
- `vpg_kurtosis`
- `vpg_p05`
- `vpg_p95`
- `apg_mean`
- `apg_std`
- `apg_iqr`
- `apg_skew`
- `apg_kurtosis`
- `apg_p05`
- `apg_p95`
- `ppg_dominant_frequency_hz`
- `ppg_high_frequency_fraction`
- `ppg_hr_bpm`
- `ppg_interval_std`
- `ppg_interval_cv`
- `ppg_amplitude_mean`
- `ppg_amplitude_std`
- `ppg_half_height_duration`
- `ppg_baseline_subtracted_area`

## PPG definitions

- mean/std: arithmetic mean and population standard deviation.
- iqr: 75th minus 25th percentile; p05/p95: 5th/95th percentile.
- skew/kurtosis: SciPy skewness and Fisher excess kurtosis with default biased estimators.
- Clean PPG: full-record NeuroKit2 ppg_clean at 125 Hz. VPG/APG: successive NumPy gradients within each window, using a 1/125-second step.
- PPG peaks: scipy.signal.find_peaks on the cleaned window, minimum spacing 41 samples; no prominence threshold.
- Heart rate: 60 divided by median peak interval in seconds. Interval std/CV use successive peak gaps.
- Pulse shape: peak-to-next-peak segments; amplitude is the left peak minus the segment minimum. Width counts samples above minimum plus half amplitude; area integrates the signal above the segment minimum. Means/std summarize segments. These are not explicit foot-to-foot pulse features.
- Frequency: Welch spectrum, 256-sample segments. Dominant frequency is the maximum-power frequency in 0.5–5 Hz. High-frequency fraction is summed 2–5 Hz power divided by summed 0.5–5 Hz power.

## ECG and cross-signal feature names

- `ecg_raw_mean`
- `ecg_raw_std`
- `ecg_raw_iqr`
- `ecg_raw_skew`
- `ecg_raw_kurtosis`
- `ecg_raw_p05`
- `ecg_raw_p95`
- `ecg_clean_mean`
- `ecg_clean_std`
- `ecg_clean_iqr`
- `ecg_clean_skew`
- `ecg_clean_kurtosis`
- `ecg_clean_p05`
- `ecg_clean_p95`
- `ecg_z_min`
- `ecg_z_max`
- `ecg_z_median`
- `ecg_z_iqr`
- `ecg_z_skew`
- `ecg_z_kurtosis`
- `ecg_dominant_frequency_hz`
- `ecg_low_power_fraction`
- `ecg_high_power_fraction`
- `ecg_rate_bpm`
- `ecg_rr_std_s`
- `ecg_rr_cv`
- `ecg_r_height_mean`
- `ecg_r_height_std`
- `ecg_ppg_peak_delay_median_s`
- `ecg_ppg_peak_delay_iqr_s`
- `ecg_ppg_peak_match_fraction`

## ECG definitions and missing values

ECG is cleaned over the full recording using NeuroKit2 at 125 Hz. Raw/cleaned statistics follow the same mean, population standard deviation, IQR, skewness, excess kurtosis and percentile definitions as PPG. The cleaned ECG is normalized within each window for ecg_z_* statistics and R-peak heights. R peaks use NeuroKit2 ecg_peaks with artifact correction disabled. Rate is 60 divided by median R–R interval in seconds; interval standard deviation/CV summarize gaps between R peaks inside the window.

ECG spectral features use Welch spectra with 256-sample segments. Dominant frequency is the maximum-power frequency in 0.5–40 Hz. Low-power fraction is 0.5–<5 Hz power divided by 0.5–40 Hz power; high-power fraction is 15–40 Hz power divided by the same total.

Cross-signal delays match each NeuroKit-detected PPG peak to its nearest preceding ECG R peak, retaining delays of 0.08–0.60 seconds. Features summarize median/IQR delay and matched-PPG-peak fraction. This provisional peak-to-peak delay is not validated pulse transit time. Its median/IQR are missing in 62.7% of all retained windows.

Missing/nonfinite feature values are imputed using training medians. No windows are dropped for missing ECG features. Input features are not standardized; targets are standardized using training statistics, and predictions are returned in mmHg. This is an input inventory, not a feature-importance ranking.
