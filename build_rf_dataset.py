
import h5py
import numpy as np
from scipy.signal import find_peaks


# -----------------------------
# SETTINGS
# -----------------------------

FS = 125
WINDOW_SECONDS = 10
WINDOW_SIZE = FS * WINDOW_SECONDS

# Select the recording and window
PART = "Part_1"
RECORD_NUMBER = 2316
WINDOW_NUMBER = 16


# -----------------------------
# LOAD RECORDING
# -----------------------------

with h5py.File(f"data/{PART}.mat", "r") as file:

    part_data = file[PART]

    # Convert recording number to zero-based index
    record_ref = part_data[RECORD_NUMBER - 1, 0]

    record = file[record_ref][:]


# -----------------------------
# EXTRACT ABP WINDOW
# -----------------------------

start = WINDOW_NUMBER * WINDOW_SIZE
end = start + WINDOW_SIZE

abp = record[start:end, 1]


# -----------------------------
# DETECT SYSTOLIC PEAKS
# -----------------------------

abp_peaks, _ = find_peaks(
    abp,
    distance=int(0.3 * FS),
    prominence=20
)


# -----------------------------
# DETECT DIASTOLIC VALLEYS
# -----------------------------

abp_valleys = []

for i in range(len(abp_peaks) - 1):

    start_index = abp_peaks[i]
    end_index = abp_peaks[i + 1]

    if end_index > start_index + 1:

        valley_index = start_index + np.argmin(
            abp[start_index:end_index]
        )

        abp_valleys.append(valley_index)


abp_valleys = np.array(abp_valleys, dtype=int)


# -----------------------------
# CALCULATE BLOOD PRESSURE
# -----------------------------

sbp = np.mean(abp[abp_peaks])
dbp = np.mean(abp[abp_valleys])


# -----------------------------
# PRINT RESULTS
# -----------------------------

print("\nRECORDING INFORMATION")
print("Recording:", PART, RECORD_NUMBER)
print("Window:", WINDOW_NUMBER)

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

print("Average SBP:", round(sbp, 2), "mmHg")
print("Average DBP:", round(dbp, 2), "mmHg")

if len(abp_peaks) > 1:

    average_interval = np.mean(np.diff(abp_peaks)) / FS

    print(
        "Average time between peaks:",
        round(average_interval, 2),
        "seconds"
    )
