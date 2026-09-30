import h5py
import numpy as np
import matplotlib.pyplot as plt

file = h5py.File("data/Part_1.mat", "r")

part1 = file["Part_1"]

# Get Record 1
record1_ref = part1[0, 0]
record1 = file[record1_ref]

# Sampling frequency
fs = 125

# Get the first 10 seconds of data
num_samples = 10 * fs

# Get PPG and ABP
ppg = record1[:num_samples, 0]
abp = record1[:num_samples, 1]

# Create time values
time = np.arange(num_samples) / fs

# Plot PPG
plt.figure()
plt.plot(time, ppg)
plt.xlabel("Time (seconds)")
plt.ylabel("PPG")
plt.title("Record 1: PPG Signal")
plt.show()

# Plot ABP
plt.figure()
plt.plot(time, abp)
plt.xlabel("Time (seconds)")
plt.ylabel("Blood Pressure (mmHg)")
plt.title("Record 1: ABP Signal")
plt.show()