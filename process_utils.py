from scipy import signal
from biosppy.signals import ecg
import torch
import numpy as np

import matplotlib.pyplot as plt

# from biosppy.signals import st


def feature_construct(data, label):
    new_data = np.zeros((12, 10))
    data_mean = np.mean(data, 0)
    time = np.arange(0, len(data_mean)) * (1.0 / 500)
    plt.plot(time, data_mean)
    plt.show()
    out = ecg.ecg(signal=data_mean, sampling_rate=500, show=True, interactive=True)
    return new_data


def singnal_fliter(data, frequency=500, highpass=45, lowpass=3):
    [b, a] = signal.butter(
        3, [lowpass / frequency * 2, highpass / frequency * 2], "bandpass"
    )
    Signal_pro = signal.filtfilt(b, a, data)
    return Signal_pro


def process_ecg(data, label):
    target_data = np.zeros((12, 250))
    # print(data.shape)
    ecg_mean = np.mean(data, axis=0)

    try:
        out_mean = ecg.ecg(
            signal=ecg_mean, sampling_rate=500, show=False, interactive=False
        )
        # filtered_mean = out_mean['filtered']
        r_pos = out_mean["rpeaks"]
    except Exception:
        print(label)
        print("1err")
        return None


    # try:
    #     q_pos, q_starts = ecg.getQPositions(out_mean, show=False)
    #     s_pos, s_ends = ecg.getSPositions(out_mean, show=False)
    # except Exception:
    #     print(label)
    #     print("2err")
    #     return None

    temp_data = np.zeros((12, 250))
    
    for c,channel in enumerate(data):
        MIN = np.min(channel)
        MAX = np.max(channel)
        if (MAX - MIN) <= 20:
            # print("bad channel")
            continue
        filtered_channel, _, _ = ecg.st.filter_signal(
        signal=channel,
        ftype="FIR",
        band="bandpass",
        order=int(0.3 * 500),
        frequency=[3, 45],
        sampling_rate=500,
        )
        count = 0
        for r in r_pos:
            if filtered_channel[r] < 40:
                continue
            t1 = max(r - 100,0)
            t2 = min(r + 150,7500)
            if t1 == 0 or t2 == 7500:
                # print("bound pass")
                continue
            MAXP = np.max(filtered_channel[t1:t2])
            MINP = np.min(filtered_channel[t1:t2])
            if (MAXP - MINP) <= 40:
                # print("bad seg")
                # print(t1,t2)
                # print(MAXP,MINP,filtered_channel[r])
                # t_ = np.arange(t1,t2)
                # plt.title("bad seg")
                # plt.plot(np.arange(0,7500),filtered_channel)
                # plt.plot(t_,filtered_channel[t1:t2],c="k")
                # plt.show()
                continue
            temp_data[c] += filtered_channel[t1:t2]
            count += 1
        if count != 0:
            temp_data[c] = temp_data[c] / count
    
    s = np.sum(temp_data,axis=0)
    if np.max(s) == 0:
        print("bad ecg with label ",label)
        return None    
    # time = np.arange(0,250)
    # for cc in temp_data:
    #     plt.plot(time,cc)
    # plt.plot(np.arange(0,7500),out_mean['filtered'])
    # plt.show()

    return temp_data


def process_ecgv2(data, label):
    # [channel] [avg, lowest, highest] [J, JX60, JX80, [JX80_seq]]
    new_data = np.zeros((12, 3, 73))
    ecg_mean = np.mean(data, 0)
    
    # plt.show()
    try:
        out_mean = ecg.ecg(
            signal=ecg_mean, sampling_rate=500, show=False, interactive=False
        )
        filtered_mean = out_mean['filtered']
        filtered_ts = out_mean['ts'] * 500
        r_pos = out_mean["rpeaks"]

    except Exception:
        print(label)
        print("1err")
        return None

    try:
        q_pos, q_starts = ecg.getQPositions(out_mean, show=False)
        s_pos, s_ends = ecg.getSPositions(out_mean, show=False)
        # p_pos, p_starts, p_ends = ecg.getPPositions(out_mean, show=True)

        # t_pos, t_starts, q_ends  = ecg.getTPositions(out_mean, show=False)
    except Exception:
        print(label)
        print("2err")
        return None

    # window_len = 20  # 40 ms
    r = 10  # 20ms half, window width is 40ms
    for c in range(0, 12):
        if np.max(data[c]) == 0:
            continue
        sampling_rate = 500
        order = int(0.3 * sampling_rate)
        filtered_channel, _, _ = ecg.st.filter_signal(
            signal=data[c],
            ftype="FIR",
            band="bandpass",
            order=order,
            frequency=[3, 45],
            sampling_rate=sampling_rate,
        )
        # filtered_channel = singnal_fliter(data[c])
        ST_J_Arr = []
        ST_JX60_Arr = []
        ST_JX80_Arr = []
        ST_JX_Arr = []
        
        # time = np.arange(0, 7500)
        # plt.plot(filtered_ts, filtered_mean, color='y', label='filtered_avg_signal')
        # plt.plot(filtered_ts, filtered_channel, color='g', label='filtered_channel_signal')

        for idx in range(0, len(r_pos)):
            t_iso = q_starts[idx] - 110  # left shift 24ms
            if t_iso < 0:
                t_iso = 0
                
            if t_iso - 20 < 0:
                t_iso = 20
            iso = np.mean(filtered_mean[t_iso - 20 : t_iso + 20])
            # print(idx,t_iso - 20 , t_iso + 20, iso)
            t_j = s_ends[idx] - 0
            # dj1 = filtered_channel[t_j] - filtered_channel[t_j+5]
            # dj2 = filtered_channel[t_j+5] - filtered_channel[t_j+10]
            
            
            ST_J = np.mean(filtered_channel[t_j : t_j + 5]) - iso
            

            t_jx1 = t_j + 30  # 60ms
            t_jx2 = t_j + 40  # 80ms

            ST_JX1 = np.mean(filtered_channel[t_jx1 - r : t_jx1 + r]) - iso
            ST_JX2 = np.mean(filtered_channel[t_jx2 - r : t_jx2 + r]) - iso
            ST_JX = filtered_channel[t_j - 10 :t_jx2 + 20] - iso

            ST_J_Arr.append(ST_J)
            ST_JX60_Arr.append(ST_JX1)
            ST_JX80_Arr.append(ST_JX2)
            ST_JX_Arr.append(ST_JX)

            # time = np.arange(t_iso - 20, t_iso + 20)
            # plt.plot(time, filtered_mean[t_iso - 20 : t_iso + 20], color="r")
            # time = np.arange(t_j - 10, t_jx2 + 20)
            # plt.plot(time, ST_JX, color="r")
        if len(ST_J_Arr) == 0:
            print(f"channel {c+1} empty")
            continue
        # avg
        new_data[c][0][0] = np.mean(ST_J_Arr)
        new_data[c][0][1] = np.mean(ST_JX60_Arr)
        new_data[c][0][2] = np.mean(ST_JX80_Arr)
        new_data[c][0][3:] = np.mean(ST_JX_Arr, axis=0)
        # lowest
        new_data[c][1][0] = np.min(ST_J_Arr)
        new_data[c][1][1] = np.min(ST_JX60_Arr)
        new_data[c][1][2] = np.min(ST_JX80_Arr)
        arg = np.argmin(np.mean(ST_JX_Arr, axis=1))
        new_data[c][1][3:] = ST_JX_Arr[arg]
        # higest
        new_data[c][2][0] = np.min(ST_J_Arr)
        new_data[c][2][1] = np.min(ST_JX60_Arr)
        new_data[c][2][2] = np.min(ST_JX80_Arr)
        arg = np.argmax(np.mean(ST_JX_Arr, axis=1))
        new_data[c][2][3:] = ST_JX_Arr[arg]

        # plt.legend(loc='upper left', bbox_to_anchor=(0, 1.0))
        # plt.title(f'channel {c+1}    label: {label}')
        # plt.show()

    return new_data


if __name__ == "__main__":
    from datasets import ecgDataset
    from torch.utils.data import DataLoader
    from biosppy.signals import ecg
    import numpy as np
    import torch

    # import os
    d = ecgDataset("./train-new.hdf5", isTrain=False)
    train_loader = DataLoader(dataset=d, batch_size=1, shuffle=False, drop_last=True)
    STE_ST1 = []
    STE_ST2 = []
    STD_ST1 = []
    STD_ST2 = []
    OT_ST1 = []
    OT_ST2 = []
    for i, data in enumerate(train_loader):
        d, label = data
        d = d[0]
        label = label[0]
        # print(d.shape)
        d[:, 1] = d[:, 1] - d[:, 0]
        d[:, 2] = d[:, 2] - d[:, 0]
        score = torch.sum(d, 0)
        st1 = score[1]
        st2 = score[2]

        if label[0] == 1:
            STE_ST1.append(st1)
            STE_ST2.append(st2)
            # print(f"STE\t ST1:{st1} ST2:{st2}")
        if label[1] == 1:
            STD_ST1.append(st1)
            STD_ST2.append(st2)
            # print(f"STD\t ST1:{st1} ST2:{st2}")
        if label[2] == 1:
            OT_ST1.append(st1)
            OT_ST2.append(st2)
            # print(f"Other\t ST1:{st1} ST2:{st2}")
        # if i % 1000 == 0:
    # print(STE_ST1)
    STE_ST1_AVG = np.mean(STE_ST1)
    STE_ST2_AVG = np.mean(STE_ST2)
    STD_ST1_AVG = np.mean(STD_ST1)
    STD_ST2_AVG = np.mean(STD_ST2)
    OT_ST1_AVG = np.mean(OT_ST1)
    OT_ST2_AVG = np.mean(OT_ST2)
    print(f"STE_ST1_AVG: {STE_ST1_AVG}\t STE_ST2_AVG: {STE_ST2_AVG}")
    print(f"STD_ST1_AVG: {STD_ST1_AVG}\t STD_ST2_AVG: {STD_ST2_AVG}")
    print(f"OT_ST1_AVG: {OT_ST1_AVG}\t OT_ST2_AVG: {OT_ST2_AVG}")
