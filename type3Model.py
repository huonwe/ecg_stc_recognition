# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from transformer import TransformerTS
from resnet import resnet50,resnet18, resnet34, resnet101

import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=12,
            out_channels=24,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.conv2 = nn.Conv1d(
            in_channels=24,
            out_channels=48,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.conv3 = nn.Conv1d(
            in_channels=48,
            out_channels=96,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm1d(12)
        self.bn1 = nn.BatchNorm1d(24)
        self.bn2 = nn.BatchNorm1d(48)
        self.bn3 = nn.BatchNorm1d(96)

    def forward(self, x):
        # x = self.bn0(x)
        x = F.relu(self.maxpool(self.conv1(x)))
        x = self.bn1(x)
        x = F.relu(self.maxpool(self.conv2(x)))
        x = self.bn2(x)
        x = F.relu(self.maxpool(self.conv3(x)))
        x = self.bn3(x)
        return x


class Type3(nn.Module):
    def __init__(self):
        super(Type3, self).__init__()
        self.cnn = CNN()
        self.lstm1 = nn.LSTM(
            input_size=96, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2
        )
        self.fc11 = nn.Linear(in_features=5 * 32, out_features=32)
        self.fc12 = nn.Linear(in_features=32, out_features=3)

        self.fc21 = nn.Linear(in_features=3, out_features=24)
        self.fc22 = nn.Linear(in_features=24, out_features=48)
        self.fc23 = nn.Linear(in_features=48, out_features=96)
        self.fc24 = nn.Linear(in_features=96, out_features=3)

    def forward(self, x):
        x_avg = x[:, :, 0, :].squeeze()  # -1, 12, 73
        x_avg_point = x_avg[:, :, :3]  # -1, 12, 3
        x_avg_st_seq = x_avg[:, :, 3:]  # -1, 12, 70

        x_low = x[:, :, 1, :].squeeze()  # -1, 12, 73
        x_low_point = x_low[:, :, :3]  # -1, 12, 3
        x_low_st_seq = x_low[:, :, 3:]  # -1, 12, 70

        x_high = x[:, :, 2, :].squeeze()  # -1, 12, 73
        x_high_point = x_high[:, :, :3]  # -1, 12, 3
        x_high_st_seq = x_high[:, :, 3:]  # -1, 12, 70

        avg_seq = self.cnn(x_avg_st_seq)  # -1, 96, 5
        avg_seq, _ = self.lstm1(avg_seq.permute(0, 2, 1))  # -1, 5, 32
        avg_seq = avg_seq.reshape(-1, 5 * 32)
        avg_seq = F.relu(self.fc11(avg_seq))
        avg_seq = self.fc12(avg_seq)  # -1, 3

        low_seq = self.cnn(x_low_st_seq)  # -1, 96, 5
        low_seq, _ = self.lstm1(low_seq.permute(0, 2, 1))  # -1, 5, 32
        low_seq = low_seq.reshape(-1, 5 * 32)
        low_seq = F.relu(self.fc11(low_seq))
        low_seq = self.fc12(low_seq)  # -1, 3

        high_seq = self.cnn(x_high_st_seq)  # -1, 96, 5
        high_seq, _ = self.lstm1(high_seq.permute(0, 2, 1))  # -1, 5, 32
        high_seq = high_seq.reshape(-1, 5 * 32)
        high_seq = F.relu(self.fc11(high_seq))
        high_seq = self.fc12(high_seq)  # -1, 3

        x_avg_point = self.fc21(x_avg_point)
        x_avg_point = self.fc22(x_avg_point)
        x_avg_point = self.fc23(x_avg_point)
        x_avg_point = self.fc24(x_avg_point)
        x_avg_point = x_avg_point.sum(dim=1)

        x_low_point = self.fc21(x_low_point)
        x_low_point = self.fc22(x_low_point)
        x_low_point = self.fc23(x_low_point)
        x_low_point = self.fc24(x_low_point)
        x_low_point = x_low_point.sum(dim=1)

        x_high_point = self.fc21(x_high_point)
        x_high_point = self.fc22(x_high_point)
        x_high_point = self.fc23(x_high_point)
        x_high_point = self.fc24(x_high_point)
        x_high_point = x_high_point.sum(dim=1)

        return x_avg_point + x_low_point + x_high_point


class Type4(nn.Module):
    def __init__(self):
        super(Type4, self).__init__()
        self.batch_size = 32
        self.attention = nn.Parameter(torch.ones((12, 70)))
        self.threhold = nn.Parameter(
            torch.zeros((12, 3, 3))
        )  # channel (ste,std,others) (jPoint, j60, j80)
        self.JRelation = nn.Parameter(torch.randn(3, 3))

    def forward(self, x):
        result = torch.zeros((32, 3)).cuda()
        # print(x.shape)
        x_avg = x[:, :, 0, :].squeeze()  # -1, 12, 73
        x_avg_point = x_avg[:, :, :3]  # -1, 12, 3
        x_avg_st_seq = x_avg[:, :, 3:]  # -1, 12, 70
        x_avg_st_seq = torch.mul(x_avg_st_seq, self.attention)

        x_low = x[:, :, 1, :].squeeze()  # -1, 12, 73
        x_low_point = x_low[:, :, :3]  # -1, 12, 3
        x_low_st_seq = x_low[:, :, 3:]  # -1, 12, 70
        x_low_st_seq = torch.mul(x_low_st_seq, self.attention)

        x_high = x[:, :, 2, :].squeeze()  # -1, 12, 73
        x_high_point = x_high[:, :, :3]  # -1, 12, 3
        x_high_st_seq = x_high[:, :, 3:]  # -1, 12, 70
        x_high_st_seq = torch.mul(x_high_st_seq, self.attention)

        batch_results = torch.zeros((32,12,3)).cuda()
        for n, single in enumerate(x_avg_point):
            results = torch.zeros((12,3)).cuda()
            for cidx, c in enumerate(single):
                # print(cidx)
                result = torch.zeros((3,3)).cuda()
                # c0 is J point
                # c1 is J60 point
                # c2 is J80 point
                # steJ
                result[0,0] = c[0] - self.threhold[cidx][0][0]
                # steJ60
                result[0,1] = c[1] - self.threhold[cidx][0][1]
                # steJ80
                result[0,2] = c[2] - self.threhold[cidx][0][2]

                # stdJ
                result[1,0] = c[0] - self.threhold[cidx][1][0]
                # stdJ60
                result[1,1] = c[1] - self.threhold[cidx][1][1]
                # stdJ80
                result[1,2] = c[2] - self.threhold[cidx][1][2]

                # otherJ
                result[2,0] = c[0] - self.threhold[cidx][2][0]
                # otherJ60
                result[2,1] = c[1] - self.threhold[cidx][2][1]
                # otherJ80
                result[2,2] = c[2] - self.threhold[cidx][2][2]
                    # [
                    #     [steJ, steJ60, steJ80],
                    #     [stdJ, stdJ60, stdJ80],
                    #     [otherJ, otherJ60, otherJ80],
                    # ]
                result = torch.mm(result, self.JRelation) # 3, 3
                result = result.sum(dim=0) # col axis sum # 3
                results[cidx,:] = result
            results = F.softmax(results, dim=1).sum(dim=0)
            results = F.sigmoid(results)
            batch_results[n,:,:] = results
        return batch_results.sum(dim=1)
    

class TIBlock(nn.Module):
    def __init__(self, sequenceLen):
        super(TIBlock, self).__init__()
        self.sequenceLen = sequenceLen
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(sequenceLen,sequenceLen) 
        self.fc2 = nn.Linear(sequenceLen,sequenceLen)
        self.dropout = nn.Dropout()
        self.layerNorm = nn.LayerNorm(sequenceLen)
    def forward(self, x):
        x1 = F.relu(x)
        x1 = self.fc1(x1)
        x1 = self.dropout(x1)
        x = self.fc2(x) + x1
        x = self.layerNorm(x)
        return x

# class TIM(nn.Module):
#     def __init__(self):
#         super(TIM, self).__init__()
#         self.block1 = TIMBlock(in_features=70)
#         self.attributes = nn.Parameter(torch.randn((64)))
        
#     def forward(self, x):
#         x = self.block1(x)
        
#         inputs = self.
        
        
        
class Type5(nn.Module):
    def __init__(self):
        super(Type5, self).__init__()
        self.batch_size = 32
        self.attention = nn.Parameter(torch.ones((1,70)))
        self.resnet = resnet18(num_classes=2)
        self.weight = nn.Parameter(torch.ones(2,3))
        self.fc1 = nn.Linear(3,36)
        self.fc2 = nn.Linear(36,2)
        self.TIDBlock = TIBlock(sequenceLen=70)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(12)
        self.bn3 = nn.BatchNorm1d(12)
        
        self.rnn = nn.GRU(
            input_size=70, hidden_size=70, num_layers=5, batch_first=True,
            dropout=0.2,bidirectional=True
        )
        
        # self.fc3 = nn.Linear(70*12*2,12*7)
        # self.fc4 = nn.Linear(12*7,2)
    def forward(self, x):
        attention = self.attention.repeat(12,1)
        # attention = F.softmax(self.attention, dim=1).repeat(12,1)
        x_avg = x[:, :, 0, :].squeeze()  # -1, 12, 73
        x_avg_point = x_avg[:, :, :3]  # -1, 12, 3
        x_avg_st_seq = self.bn1(x_avg[:, :, 3:])  # -1, 12, 70
        x_avg_st_seq = torch.mul(x_avg_st_seq, attention)
        x_low = x[:, :, 1, :].squeeze()  # -1, 12, 73
        x_low_point = x_low[:, :, :3]  # -1, 12, 3
        x_low_st_seq = self.bn2(x_low[:, :, 3:])  # -1, 12, 70
        x_low_st_seq = torch.mul(x_low_st_seq, attention)

        x_high = x[:, :, 2, :].squeeze()  # -1, 12, 73
        x_high_point = x_high[:, :, :3]  # -1, 12, 3
        x_high_st_seq = self.bn3(x_high[:, :, 3:])  # -1, 12, 70
        x_high_st_seq = torch.mul(x_high_st_seq, attention)
        
        x_avg_st_seq = self.TIDBlock(x_avg_st_seq)
        x_low_st_seq = self.TIDBlock(x_low_st_seq)
        x_high_st_seq = self.TIDBlock(x_high_st_seq)
        
        r1 = self.resnet(x_avg_st_seq)
        r2 = self.resnet(x_low_st_seq)
        r3 = self.resnet(x_high_st_seq)
        # r = self.fc()
        weight = F.softmax(self.weight, dim=1)
        r1 = weight[0][0]*r1 + weight[0][1]*r2 + weight[0][2]*r3
        # print(r1.shape)
        # r1 = torch.flatten(r1,start_dim=1)
        # result = F.relu(self.fc3(r1))
        # result = F.sigmoid(self.fc4(result))
        xa = self.fc1(x_avg_point)
        xb = self.fc1(x_low_point)
        xc = self.fc1(x_high_point)
        # self.weight[0]*
        xa = self.fc2(xa)
        xb = self.fc2(xb)
        xc = self.fc2(xc)

        xa = torch.max(xa, 1)
        xb = torch.max(xb, 1)
        xc = torch.max(xc, 1)
        # # print(xa.values.shape)
        r2 = weight[1][0]*xa.values + weight[1][1]*xb.values + weight[1][2]*xc.values
        
        pred = r1 + r2
        # result = torch.zeros((32,2)).cuda()
        
        # result[:,0] = pred[:,0] / F.sigmoid(pred[:,2])
        # result[:,1] = pred[:,1] / F.sigmoid(pred[:,2])
        # result[:,2] = pred[:,2] / F.sigmoid(pred[:,0]+pred[:,1])
        # result = F.sigmoid(result)
        # binary = torch.zeros((32,2)).cuda()
        # binary[:,0] = pred[:,1]+pred[:,0]
        # binary[:,1] = pred[:,2]
        # # print(binary.shape)
        # binary = F.softmax(binary, dim=-1)
        
        # result = torch.zeros((32,3)).cuda()
        # result[:,2] = binary[:,1]
        # pred[:,1] = F.sigmoid(pred[:,1])
        # pred[:,0] = F.sigmoid(pred[:,2])
        # result[:,1] =torch.mul(pred[:,1],binary[:,0])
        # result[:,0] =torch.mul(pred[:,0],binary[:,0])

        
        # result = torch.cat((binary[:,1],))

        
        
        return pred
    def initialize(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight.data)

class Splite(nn.Module):
    def __init__(self):
        super(Splite, self).__init__()
        self.batch_size = 32
        self.attention = nn.Parameter(torch.ones((1,70)))
        self.resnet = resnet34(num_classes=1)
        self.resnet2 = resnet34(num_classes=2)
        self.weight = nn.Parameter(torch.ones(2,3))
        self.fc1 = nn.Linear(3,36)
        self.fc2 = nn.Linear(36,3)
        self.TIDBlock = TIBlock(sequenceLen=70)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(12)
        self.bn3 = nn.BatchNorm1d(12)
    def forward(self, x):
        attention = self.attention.repeat(12,1)
        # attention = F.softmax(self.attention, dim=1).repeat(12,1)
        x_avg = x[:, :, 0, :].squeeze()  # -1, 12, 73
        x_avg_point = x_avg[:, :, :3]  # -1, 12, 3
        x_avg_st_seq = self.bn1(x_avg[:, :, 3:])  # -1, 12, 70
        x_avg_st_seq = torch.mul(x_avg_st_seq, attention)
        x_low = x[:, :, 1, :].squeeze()  # -1, 12, 73
        x_low_point = x_low[:, :, :3]  # -1, 12, 3
        x_low_st_seq = self.bn2(x_low[:, :, 3:])  # -1, 12, 70
        x_low_st_seq = torch.mul(x_low_st_seq, attention)

        x_high = x[:, :, 2, :].squeeze()  # -1, 12, 73
        x_high_point = x_high[:, :, :3]  # -1, 12, 3
        x_high_st_seq = self.bn3(x_high[:, :, 3:])  # -1, 12, 70
        x_high_st_seq = torch.mul(x_high_st_seq, attention)
        
        x_avg_st_seq = self.TIDBlock(x_avg_st_seq)
        x_low_st_seq = self.TIDBlock(x_low_st_seq)
        x_high_st_seq = self.TIDBlock(x_high_st_seq)
        
        r11 = self.resnet(x_avg_st_seq)
        r12 = self.resnet(x_low_st_seq)
        r13 = self.resnet(x_high_st_seq)
        r21 = self.resnet2(x_avg_st_seq)
        r22 = self.resnet2(x_low_st_seq)
        r23 = self.resnet2(x_high_st_seq)
        # r = self.fc()
        weight = F.softmax(self.weight, dim=1)
        res_ans1 = weight[0][0]*r11 + weight[0][1]*r12 + weight[0][2]*r13
        res_ans2 = weight[1][0]*r21 + weight[1][1]*r22 + weight[1][2]*r23
        
        res_ans1 = F.sigmoid(res_ans1)
        # res_ans2 = F.sigmoid(res_ans2)
        
        ans = torch.zeros((32, 3)).cuda()
        ans[:,2] = res_ans1[:,0]
        ans[:,1] = F.sigmoid(torch.mul(res_ans2[:,1] , torch.reciprocal(res_ans1[:,0])))
        ans[:,0] = F.sigmoid(torch.mul(res_ans2[:,0] , torch.reciprocal(res_ans1[:,0])))
        
        return ans

class Type6(nn.Module):
    def __init__(self):
        super(Type6, self).__init__()
        self.batch_size = 32
        # self.attention = nn.Parameter(torch.ones((1,150)))
        self.resnet = resnet50()
        self.lstm = nn.LSTM(
            input_size=12, hidden_size=32, num_layers=3,
            dropout=0.2
        )
        # 7500 32 12
        self.fc = nn.Linear(32*7500,3)
        # self.fc2 = nn.Linear(3*75, 3)
    def forward(self, x):
        x1 = self.resnet(x)
        x2,_ = self.lstm(x.permute(2,0,1)) # 7500 32 12 -> 7500 32 32
        x2 = torch.flatten(x2.permute(1,2,0),start_dim=1)
        x2 = self.fc(x2)
        # x2 = self.fc2(x2)
        
        return x1 + x2

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                
                
class Type7(nn.Module):
    def __init__(self):
        super(Type7, self).__init__()
        self.batch_size = 32
        # self.attention = nn.Parameter(torch.ones((1,150)))
        self.resnet = resnet18()
        self.lstm = nn.LSTM(
            input_size=12, hidden_size=32, num_layers=3,
            dropout=0.2
        )
        self.fc = nn.Linear(32*150,3*75)
        self.fc2 = nn.Linear(3*75, 3)
        self.tsf = TransformerTS(input_dim=150)
        self.at = nn.MultiheadAttention(12,6)
    def forward(self, x):
        at_out,_ = self.at(x.permute(2,0,1),x.permute(2,0,1),x.permute(2,0,1))
        at_out = at_out.permute(1,2,0)
        self.attentioned = at_out
        
        x1 = self.resnet(at_out)
        # x2,_ = self.lstm(x.permute(2,0,1)) # 7500 32 12 -> 7500 32 32
        # x2 = torch.flatten(x2.permute(1,2,0),start_dim=1)
        # x2 = F.relu(self.fc(x2))
        # x2 = F.relu(self.fc2(x2))
        
        x3 = self.tsf(x.permute(0,2,1))
        
        return x1 + x3

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
# . 数据集平衡
# . 移动数据集1,2
# . Resnet50

class STC(nn.Module):
    def __init__(self):
        super(STC, self).__init__()
        self.batch_size = 32
        # self.attention = nn.Parameter(torch.ones((1,150)))
        self.resnet = resnet50(num_classes=2)
        
    def forward(self, x):
        x = self.resnet(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                