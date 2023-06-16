# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import resnet50, resnet18, resnet34
class CNN_(nn.Module):
    def __init__(self, num_channel):
        super(CNN_, self).__init__()
        self.num_channel = num_channel
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel)
        self.bn2 = nn.BatchNorm1d(num_features=self.num_channel*2)
        self.bn3 = nn.BatchNorm1d(num_features=self.num_channel*4)
        self.bn4 = nn.BatchNorm1d(num_features=self.num_channel*8)
        self.bn5 = nn.BatchNorm1d(num_features=self.num_channel*16)
        self.bn6 = nn.BatchNorm1d(num_features=self.num_channel*32)
        self.bn7 = nn.BatchNorm1d(num_features=self.num_channel*64)

        self.conv1 = nn.Conv1d(
            in_channels=self.num_channel,
            out_channels=self.num_channel * 2,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.num_channel * 2,
            out_channels=self.num_channel * 4,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.conv3 = nn.Conv1d(
            in_channels=self.num_channel * 4,
            out_channels=self.num_channel * 8,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.conv4 = nn.Conv1d(
            in_channels=self.num_channel * 8,
            out_channels=self.num_channel * 16,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.conv5 = nn.Conv1d(
            in_channels=self.num_channel * 16,
            out_channels=self.num_channel * 32,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.conv6 = nn.Conv1d(
            in_channels=self.num_channel * 32,
            out_channels=self.num_channel * 64,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.maxpool = nn.MaxPool1d(2,2)

    def forward(self, x):
        # print(f"bn:{x.shape}")
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.bn3(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.bn4(x)
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.bn5(x)
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = self.bn6(x)
        x = self.maxpool(x)
        x = F.relu(self.conv6(x))
        x = self.bn7(x)
        x = self.maxpool(x)

        return x


class model7500(nn.Module):
    def __init__(self):
        super(model7500, self).__init__()
        self.resnet = resnet50()
        # self.lstm = nn.LSTM(
        #     input_size=12, hidden_size=24, num_layers=3,
        #     dropout=0.3
        # )
        # self.fc1 = nn.Linear(24*250,250)
        # self.fc2 = nn.Linear(250, 3)
        # self.at0 = nn.MultiheadAttention(12,4,dropout=0.5)
        # self.at = nn.MultiheadAttention(24,2,dropout=0.5)
    def forward(self, x):
        # x1,_ = self.at0(x.permute(2,0,1),x.permute(2,0,1),x.permute(2,0,1))
        
        # # x1 = x.permute(2,0,1)
        # x1,_ = self.lstm(x1)

        # x1,_ = self.at(x1,x1,x1)
        # self.attentioned = x1.permute(1,2,0)
        # x1 = torch.flatten(x1.permute(1,2,0),start_dim=1)
        # x1 = F.relu(self.fc1(x1))
        # x1 = self.fc2(x1)
        x3 = self.resnet(x)
        
        mix = x3
        # batch_size, 3
        # print(mix_min.shape)
        min_v, _ = torch.max(mix[:,:2],dim=-1)
        mix[:,2] -= min_v
        
        
        return mix