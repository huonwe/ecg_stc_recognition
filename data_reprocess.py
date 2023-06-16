import os
import numpy as np
import h5py
from create_hdf5 import HDF5DatasetWriter
import scipy.io as scio
import process_utils

from biosppy.signals import ecg

import pandas as pd

df = pd.read_excel("./data/Train.xlsx","Sheet1")
# print(df['name'])

# for idx, name, ste, std, random in df:
#     print(name)

train_ecgs = h5py.File("./val.hdf5", "r")

# train-set 4 1err
# val-set 1 2err

# print(train_eegs)

writer = HDF5DatasetWriter(dims=[800, 12, 3, 73], outputPath="val-new-3.hdf5")
# writer2 = HDF5DatasetWriter(dims=[3200 - 4, 12, 3, 43], outputPath="train-new3.hdf5")

index = 0
count_t = 0
try:
    target: list
    for idx, ecg_label in enumerate(train_ecgs["labels"]):
        data = train_ecgs["data"][idx]  # 12,7500
        # path = 
        # data = scio.loadmat("./data/Train/"+path)
        new_data = None
        if ecg_label[0] == 1:
            new_data = process_utils.process_ecgv2(data, ecg_label)
        if type(new_data) == type(None):
            continue
        # writer.add(new_data, ecg_label)
        index += 1
        print(index)
# except Exception:
#     print("err happen")
finally:
    print(index)
    writer.close()
