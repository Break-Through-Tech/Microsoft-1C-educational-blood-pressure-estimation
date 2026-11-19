# AdaBoost: PPG + ECG feature inventory

68 inputs (37 PPG + 31 ECG/cross-signal). ABP supplies labels only.

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

Full PPG and ECG definitions are in [the shared feature documentation](../features.md). ECG uses raw/clean statistics, normalized shape, R-peak timing/heights, Welch power, and nearest-preceding-R-to-PPG-peak delays within 0.08–0.60 seconds. These delays are not validated pulse transit time. Missing ECG stays missing and is imputed from training medians; no rows are dropped. Input features are not standardized; targets are standardized using training statistics and predictions returned in mmHg.

Delay median/IQR are missing in 62.7% of all retained windows under the provisional matching bounds; this limits interpretation of these particular features.
