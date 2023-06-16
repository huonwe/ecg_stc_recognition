# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from model_paper import SelfAttention

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

# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention, self).__init__()
#         self.w = torch.nn.Parameter(torch.randn(500,500))
        
#     def forward(self,x):
#         M = F.tanh(x)
#         # print(M.shape)
#         A = F.softmax(torch.mm(M,self.w))

class SplitNet(nn.Module):
    def __init__(self, batch_size):
        super(SplitNet, self).__init__()
        self.batch_size = batch_size
        self.num_channel = 12
        
        self.cnn_ = CNN_(num_channel=12) # after lstm
        
        self.conv0 = nn.Conv1d(
            in_channels=12,
            out_channels=12,
            kernel_size=3,
            padding=1,
            padding_mode='replicate'
        )
        
        self.lstm1 = nn.LSTM(
            input_size=12,
            hidden_size=6,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        self.conv1 = nn.Conv1d(
            in_channels=6,
            out_channels=3,
            kernel_size=3,
            padding=1,
            padding_mode='replicate'
        )
        self.maxpool = nn.MaxPool1d(2,1)
        
        # self.conv31 = nn.Conv1d(in_channels=12*16)
        
        self.linear1 = nn.Linear(in_features=192*468,out_features=500)
        self.linear2 = nn.Linear(in_features=500,out_features=2)
        
        self.linear3 = nn.Linear(in_features=3 * 7499, out_features=500)
        self.linear4 = nn.Linear(in_features=500,out_features=1)
        
        self.linear5 = nn.Linear(in_features=500,out_features=3)
        
        self.channelW = nn.Parameter(torch.randn(12,12))
        
        self.bn1 = nn.BatchNorm1d(6)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = torch.matmul(x, self.channelW)
        # print(x1.shape)
        
        x1, _ = self.lstm1(x) # N,Hin,L -> N,L,Hin
        x1 = x1.permute(0,2,1)
        # print(x1.shape) # -1, 6, 7500
        x1 = self.bn1(x1)
        x1 = self.maxpool(self.conv1(x1))  # -1, 3, 7499
        x1 = x1.view(-1,3 * 7499)
        x1 = F.relu(self.linear3(x1))
        # print(x1.shape)

        # print(x.shape)
        x = x.permute(0,2,1)
        x2 = self.cnn_(x)
        # print(x2.shape) # -1, 192, 468
        x2 = x2.view(-1,192*468)
        x2 = self.linear1(x2)
        
        x1 = x1 - x2

        x2 = self.linear2(x2)
        
        x1 = self.linear4(x1)
        x = torch.cat((x2,x1),dim=1)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)
            

class SplitNetV2(nn.Module):
    def __init__(self, batch_size):
        super(SplitNetV2, self).__init__()
        self.batch_size = batch_size
        self.num_channel = 12
        
        self.cnn_ = CNN_(num_channel=12) # after lstm
        
        self.conv0 = nn.Conv1d(
            in_channels=12,
            out_channels=12,
            kernel_size=3,
            padding=1,
            padding_mode='replicate'
        )
        
        self.lstm1 = nn.LSTM(
            input_size=12,
            hidden_size=6,
            num_layers=3,
            dropout=0.3,
            batch_first=True
        )
        self.conv1 = nn.Conv1d(
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            padding=1,
            padding_mode='replicate'
        )
        self.conv2 = nn.Conv1d(
            in_channels=12,
            out_channels=24,
            kernel_size=3,
            padding=1,
            padding_mode='replicate'
        )
        self.maxpool = nn.MaxPool1d(2,1)
                
        self.linear1 = nn.Linear(in_features=768*117,out_features=500)
        self.linear2 = nn.Linear(in_features=500,out_features=2)
        
        self.linear3 = nn.Linear(in_features=24 * 7498, out_features=500)
        self.linear4 = nn.Linear(in_features=500,out_features=1)
        
        self.linear5 = nn.Linear(in_features=500,out_features=3)
        
        self.channelW = nn.Parameter(torch.randn(12,12))
        
        self.bn1 = nn.BatchNorm1d(6)
        self.bn2 = nn.BatchNorm1d(12)
        self.bn3 = nn.BatchNorm1d(24)
        # self.attention = SelfAttention(2,12,12,0.5)
        
        # self.omiga = nn.Parameter(torch.randn(1,500))
        
    def forward(self, x):
        x = x.permute(0,2,1)
        # x = self.attention(x)
        x = torch.matmul(x, self.channelW)
        # print(x.shape)
        x = self.conv0(x.permute(0,2,1))
        x1, _ = self.lstm1(x.permute(0,2,1)) # N,Hin,L -> N,L,Hin
        x1 = x1.permute(0,2,1)
        # print(x1.shape) # -1, 6, 7500
        x1 = self.bn1(x1)
        x1 = F.relu(self.maxpool(self.conv1(x1)))  # -1, 12, 7499
        x1 = self.bn2(x1)
        x1 = F.relu(self.maxpool(self.conv2(x1)))  # -1, 24, 7499
        # print(x1.shape)
        x1 = self.bn3(x1)
        x1 = x1.view(-1,24 * 7498)
        x1 = F.relu(self.linear3(x1))
        # print(x1.shape)

        # print(x.shape)
        # print(x.shape)
        x2 = self.cnn_(x)
        # print(x2.shape) # -1, 192, 468
        x2 = x2.view(-1,768*117)
        x2 = F.relu(self.linear1(x2))
        
        x1 = x1 - x2
        
        x2 = self.linear2(x2)
        x1 = self.linear4(x1)
        x = torch.cat((x2,x1),dim=1)
        x = F.sigmoid(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class SplitNetV3(nn.Module):
    def __init__(self, batch_size):
        super(SplitNetV3, self).__init__()
        self.batch_size = batch_size
        
        self.fc0 = nn.Linear(in_features=24*4, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        
        self.conv0 = nn.Conv1d(in_channels=12, out_channels=24,kernel_size=1,padding=0)
        # self.conv1 = nn.Conv1d(in_channels=24, out_channels=48,kernel_size=3,padding=0)

        # self.fc1 = nn.Linear(in_features=4, out_features=64)
        # self.conv1 = nn.Conv1d(in_channels=12, out_channels=1,kernel_size=3,padding=0)
        
        self.fc2 = nn.Linear(in_features=32, out_features=3)
        
        self.maxpool = nn.MaxPool1d(3)
    def forward(self, x):
        # print(x.shape)
        x[:,:,1] = x[:,:,1] - x[:,:,0]
        x[:,:,2] = x[:,:,2] - x[:,:,0]
        x = self.conv0(x)
        # print(x.shape)
        x = x.view(-1,24*4)
        x = self.fc0(x)
        x = self.fc1(x)
        
        
        # x = self.maxpool(self.conv1(x))
        # print(x.shape)
        # x = x.view(-1, 24*10)
        x = F.sigmoid(self.fc2(x))
        
        return x
## 共享权重后，残差结构应当取消，



    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.tf1 = nn.Transformer()
        
    def forward(self, x):
        x = self.tf1(x)
        print(x.shape)
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)