import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk

MAT_FILE = "data/Part_1.mat"

SAMPLE_RATE = 125
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS   # 625 samples


def get_bp_labels(abp_window):
    """
    Extract systolic and diastolic blood pressure
    from one 5-second ABP window.
    """

    # Find systolic peaks
    peak_indices, _ = find_peaks(
        abp_window,
        distance=SAMPLE_RATE * 0.4
    )

    # Find diastolic valleys by finding peaks
    # in the inverted ABP signal
    valley_indices, _ = find_peaks(
        -abp_window,
        distance=SAMPLE_RATE * 0.4
    )

    if len(peak_indices) == 0 or len(valley_indices) == 0:
        return None

    systolic_values = abp_window[peak_indices]
    diastolic_values = abp_window[valley_indices]

    # Basic physiological filtering
    systolic_values = systolic_values[
        (systolic_values >= 70) &
        (systolic_values <= 220)
    ]

    diastolic_values = diastolic_values[
        (diastolic_values >= 30) &
        (diastolic_values <= 140)
    ]

    if len(systolic_values) == 0 or len(diastolic_values) == 0:
        return None

    # Use median so one weird heartbeat does not
    # dominate the label
    sbp = np.median(systolic_values)
    dbp = np.median(diastolic_values)

    # Sanity check
    if sbp <= dbp:
        return None
    if sbp - dbp < 10: # Do not include records where pulse is less than 10mmHg
        return None

    return sbp, dbp


X = []
y = []
record_ids = []

with h5py.File(MAT_FILE, "r") as f:

    part = f["Part_1"]

    for record_id in range(part.shape[0]):

        ref = part[record_id, 0]
        data = f[ref][:]

        ppg = data[:, 0]
        abp = data[:, 1]

        # Chop the recording into non-overlapping
        # 5-second windows
        for start in range(0, len(ppg) - WINDOW_SIZE + 1, WINDOW_SIZE):

            end = start + WINDOW_SIZE

            ppg_window = ppg[start:end]
            abp_window = abp[start:end]

            # Make sure there are exactly 625 samples
            if len(ppg_window) != WINDOW_SIZE:
                continue

            labels = get_bp_labels(abp_window)

            if labels is None:
                continue

            sbp, dbp = labels

            X.append(ppg_window)
            y.append([sbp, dbp])
            record_ids.append(record_id)


X = np.array(X)
y = np.array(y)
record_ids = np.array(record_ids)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("SBP min/max:", y[:, 0].min(), y[:, 0].max())
print("DBP min/max:", y[:, 1].min(), y[:, 1].max())

print("SBP mean:", y[:, 0].mean())
print("DBP mean:", y[:, 1].mean())

print("\nFirst 20 labels:")
print(y[:20])

unique_records, counts = np.unique(record_ids, return_counts=True)

print("Number of records used:", len(unique_records))
print("Min windows per record:", counts.min())
print("Max windows per record:", counts.max())
print("Mean windows per record:", counts.mean())

high_sbp_idx = np.where(y[:, 0] > 180)[0]
high_dbp_idx = np.where(y[:, 1] > 110)[0]

print("SBP > 180:", len(high_sbp_idx))
print("DBP > 110:", len(high_dbp_idx))

print("\nSome high SBP labels:")
print(y[high_sbp_idx[:10]])

print("\nSome high DBP labels:")
print(y[high_dbp_idx[:10]])


# Create plots using neurokit2. Load the data while the file is open.
with h5py.File(MAT_FILE, "r") as f:
    recordings = f["Part_1"]
    first_reference = recordings[0, 0]
    record = np.array(f[first_reference])

ppg = record[:2000, 0]
signals, info = nk.ppg_process(ppg, sampling_rate=125)

epochs = nk.epochs_create(signals, events=[250, 500, 700, 800], sampling_rate=125,
                         epochs_start=-0.1, epochs_end=1.9)
nk.epochs_plot(epochs)

#analyze_epochs = nk.ppg_analyze(epochs, sampling_rate=125)
nk.ppg_plot(signals, info)
plt.show()
