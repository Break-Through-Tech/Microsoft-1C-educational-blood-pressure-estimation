import h5py
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# Open the MATLAB file
file = h5py.File("data/Part_1.mat", "r")

# Access Part 1
part1 = file["Part_1"]

# Get Record 1
record1_ref = part1[0, 0]
record1 = file[record1_ref]

# Sampling frequency
fs = 125

# Get the first 10 seconds of the PPG signal
num_samples = 10 * fs
ppg = record1[:num_samples, 0]

print("Number of PPG samples:", len(ppg))
print("First 10 PPG values:")
print(ppg[:10])

# Process the PPG signal with NeuroKit2
signals, info = nk.ppg_process(ppg, sampling_rate=fs)

print("\nNeuroKit2 columns:")
print(signals.columns)

print("\nNumber of detected PPG peaks:")
print(len(info["PPG_Peaks"]))

# Plot the processed PPG
nk.ppg_plot(signals, info)

plt.show()