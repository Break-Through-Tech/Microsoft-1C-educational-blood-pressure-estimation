
import h5py
import numpy as np
from scipy.signal import find_peaks

# Settings
FS = 125
WINDOW_SECONDS = 10
WINDOW_SIZE = FS * WINDOW_SECONDS

PART = "Part_4"
RECORD_NUMBER = 2717
WINDOW_NUMBER = 0

# Load recording
with h5py.File(f"data/{PART}.mat", "r") as file:
    part_data = file[PART]
    record_ref = part_data[RECORD_NUMBER - 1, 0]
    record = file[record_ref][:]

# Extract ABP window
start = WINDOW_NUMBER * WINDOW_SIZE
end = start + WINDOW_SIZE
abp = record[start:end, 1]

# Detect systolic peaks
abp_peaks, _ = find_peaks(
    abp,
    distance=int(0.3 * FS),
    prominence=20
)

# Detect one valley between consecutive peaks
abp_valleys = []

for i in range(len(abp_peaks) - 1):
    start_index = abp_peaks[i]
    end_index = abp_peaks[i + 1]

    valley_index = start_index + np.argmin(
        abp[start_index:end_index]
    )
    abp_valleys.append(valley_index)

abp_valleys = np.array(abp_valleys, dtype=int)

print("\nSYSTOLIC PEAKS")
for i, index in enumerate(abp_peaks):
    print(
        f"Peak {i + 1}: "
        f"Time = {index / FS:.2f}s, "
        f"ABP = {abp[index]:.2f} mmHg"
    )

print("\nDIASTOLIC VALLEYS")
for i, index in enumerate(abp_valleys):
    print(
        f"Valley {i + 1}: "
        f"Time = {index / FS:.2f}s, "
        f"ABP = {abp[index]:.2f} mmHg"
    )

print("\nSUMMARY")
print("Number of peaks:", len(abp_peaks))
print("Number of valleys:", len(abp_valleys))
print("Average SBP:", round(np.mean(abp[abp_peaks]), 2))
print("Average DBP:", round(np.mean(abp[abp_valleys]), 2))
print(
    "Average time between peaks:",
    round(np.mean(np.diff(abp_peaks)) / FS, 2),
    "seconds"
)