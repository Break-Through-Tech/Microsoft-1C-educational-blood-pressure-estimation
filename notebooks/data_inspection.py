import h5py
import numpy as np

with h5py.File("data/Part_1.mat", "r") as f:
    data = f["Part_1"]

    # Get the first record from the MATLAB cell array
    ref = data[0, 0]
    record = np.array(f[ref])

    print("Shape:", record.shape)
    print(record[:5])