#encoding=utf8
import h5py
import os
import numpy as np
import scipy.io as scio


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
    # trainWriter = HDF5DatasetWriter(dims=[3200,12,15*500],outputPath='./train.hdf5')
    # ValWriter = HDF5DatasetWriter(dims=[800,12,15*500],outputPath='./val.hdf5')
    TestWriter = HDF5DatasetWriter(dims=[2000,12,15*500],outputPath='./check.hdf5')
    try:
        for root,dirs,files in os.walk('./data/CHECK'):
            for mat in files:
                index += 1
                if index % 50 == 0:
                    print("index: %s" % index)
                path = os.path.join(root, mat)
                data = scio.loadmat(path)
                # print(data['ecg'])
                # os._exit(0)
                # data = h5py.File(path)
                # print(data['label'][0])
                # label = [data['label'][0][0][0][0], data['label'][0][1][0][0], data['label'][0][2][0][0]]
                # label_a = np.array(label)
                # print("label: %s" % label_a)
                name = int(mat.split(".")[0])
                TestWriter.add(data['ecg'],name)
    finally:
        TestWriter.close()
