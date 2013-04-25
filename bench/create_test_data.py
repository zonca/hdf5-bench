import h5py
import numpy as np

folder = "testdata/"

total_samples = 100000

thin_data_dtype = [("L", np.int64)] + [("D%d" % i, np.double) for i in range(4)]
thin_data_thickness = len(thin_data_dtype)

thin_data = np.zeros(total_samples/thin_data_thickness, dtype = thin_data_dtype)
thin_data["D0"] = np.arange(len(thin_data))

with h5py.File(folder + "thin.h5", "w") as thin_dataset:
    thin_dataset.create_dataset("data", data=thin_data)
