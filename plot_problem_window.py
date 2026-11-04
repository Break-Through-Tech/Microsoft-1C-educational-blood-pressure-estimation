import h5py
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks

# Sampling rate and window settings
FS = 125
WINDOW_SECONDS = 10
WINDOW_SIZE = FS * WINDOW_SECONDS

# Select the recording and window we want to inspect
PART = "Part_4"
RECORD_NUMBER = 2717
WINDOW_NUMBER = 0

# Open the MATLAB dataset
with h5py.File(f"data/{PART}.mat", "r") as file:
    part_data = file[PART]

    # Retrieve the selected recording
    record_ref = part_data[RECORD_NUMBER, 0]
    record = file[record_ref][:]

# Extract the PPG and ABP signals
start = WINDOW_NUMBER * WINDOW_SIZE
end = start + WINDOW_SIZE

ppg = record[start:end, 0]
abp = record[start:end, 1]

# Create the time axis
time = np.arange(WINDOW_SIZE) / FS

# Detect PPG peaks using NeuroKit2
ppg_signals, ppg_info = nk.ppg_process(
    ppg,
    sampling_rate=FS
)

ppg_peaks = ppg_info["PPG_Peaks"]

# Detect ABP peaks and valleys
abp_peaks, _ = find_peaks(
    abp,
    distance=int(0.3 * FS),
    prominence=20
)

# Find the lowest ABP point between consecutive systolic peaks

abp_valleys = []

for i in range(len(abp_peaks) - 1):
    start = abp_peaks[i]
    end = abp_peaks[i + 1]

    if end > start + 1:
        valley_index = start + np.argmin(abp[start:end])
        abp_valleys.append(valley_index)

abp_valleys = np.array(abp_valleys, dtype=int)

# Calculate average SBP and DBP for this window
sbp = np.mean(abp[abp_peaks])
dbp = np.mean(abp[abp_valleys])

print("Average SBP:", round(sbp, 2), "mmHg")
print("Average DBP:", round(dbp, 2), "mmHg")

# Print peak counts
print("Recording:", PART, RECORD_NUMBER)
print("Window:", WINDOW_NUMBER)
print("PPG peak count:", len(ppg_peaks))
print("ABP peak count:", len(abp_peaks))
print("ABP valley count:", len(abp_valleys))

# Plot the PPG signal
plt.figure(figsize=(14, 5))
plt.plot(time, ppg, label="PPG Signal")
plt.scatter(
    time[ppg_peaks],
    ppg[ppg_peaks],
    color="red",
    label="Detected PPG Peaks"
)
plt.title("PPG Signal with Detected Peaks")
plt.xlabel("Time (seconds)")
plt.ylabel("PPG Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot the ABP signal
plt.figure(figsize=(14, 5))
plt.plot(time, abp, label="ABP Signal")
plt.scatter(
    time[abp_peaks],
    abp[abp_peaks],
    color="red",
    label="Detected ABP Peaks"
)
plt.scatter(
    time[abp_valleys],
    abp[abp_valleys],
    color="green",
    label="Detected ABP Valleys"
)
plt.title("ABP Signal with Detected Peaks and Valleys")
plt.xlabel("Time (seconds)")
plt.ylabel("Blood Pressure (mmHg)")
plt.legend()
plt.grid()
plt.show()