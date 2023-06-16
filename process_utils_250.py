from scipy import signal
from biosppy.signals import ecg
import torch
import numpy as np

import matplotlib.pyplot as plt
# from data_balance_250 import HDF5DatasetWriter

def process_ecg_writer(data, label, writer):
    ecg_mean = np.mean(data, axis=0)
    try:
        out_mean = ecg.ecg(
            signal=ecg_mean, sampling_rate=500, show=False, interactive=False
        )
        r_pos = out_mean["rpeaks"]
    except Exception:
        print(label)
        print("1err")
        return 0
    filtered_channels = np.zeros((12, 7500))
    for c, channel in enumerate(data):
        filtered_channels[c,:],_,_ = ecg.st.filter_signal(
            signal=channel,
            ftype="FIR",
            band="bandpass",
            order=int(0.3 * 500),
            frequency=[3, 45],
            sampling_rate=500,
        )
    count = 0
    for r in r_pos:
        
        t1 = max(r - 100, 0)
        t2 = min(r + 150, 7500)
        if t1 == 0 or t2 == 7500:
            continue
        temp_data = np.zeros((12, 250))
        for c, channel in enumerate(data):
            filtered_channel = filtered_channels[c]
            if filtered_channel[r] < 40:
                continue
            MIN = np.min(channel)
            MAX = np.max(channel)
            if (MAX - MIN) <= 20:
                continue
            MAXP = np.max(filtered_channel[t1:t2])
            MINP = np.min(filtered_channel[t1:t2])
            if (MAXP - MINP) <= 40:
                continue
            temp_data[c] += filtered_channel[t1:t2]
        writer.add(temp_data, label)
        count += 1
    return count
