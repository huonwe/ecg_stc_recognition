#encoding=utf8
import h5py
import os
import numpy as np
import scipy.io as scio
import pandas as pd

from process_utils import process_ecgv2, process_ecg
from process_utils_250 import process_ecg_writer

class HDF5DatasetWriter:
    def __init__(self, dims=None, outputPath=None, bufSize=100):
        assert dims != None and outputPath != None
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("data", dims, dtype="float32")
        self.name = self.db.create_dataset("name", [dims[0]], dtype="int")
        self.bufSize = bufSize
        self.buffer = {"data": [], "name": []}
        self.idx = 0  # 用来进行计数

    def add(self, rows, name):
        self.buffer["data"].extend([rows])
        self.buffer["name"].extend([name])
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.name[self.idx:i] = self.buffer["name"]
        self.idx = i
        self.buffer = {"data": [], "name": []}

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()

if __name__ =='__main__':
    index = 0
    TestWriter = HDF5DatasetWriter(dims=[2000,12,250],outputPath='./check-250-mean.hdf5')
    # df = pd.read_excel("./data/Train.xlsx","Sheet1")
    ecgs = h5py.File("check.hdf5", "r")
    try:
        for i, name in enumerate(ecgs["name"]):
            index += 1
            ecg = ecgs["data"][i]
            new_data = process_ecg(ecg, name)
            if type(new_data) == type(None):
                print("ERR TO PROCESS ",i)
                continue
            TestWriter.add(new_data,name)
            if i % 50 == 0:
                print(i)
    finally:
        TestWriter.close()
        print(index)
    