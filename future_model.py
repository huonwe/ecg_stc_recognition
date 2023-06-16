# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class CNN(nn.Module):
    def __init__(self, num_channel, dropout):
        super(CNN, self).__init__()
        self.num_channel = num_channel
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel)
        self.bn2 = nn.BatchNorm1d(num_features=self.num_channel*32)
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

        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_features=384 * 234, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=3)
        self.sigmod1 = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = self.bn2(x)
        x = x.reshape(-1, 384 * 234)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.sigmod1(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1


class CNN_(nn.Module):
    def __init__(self, num_channel):
        super(CNN_, self).__init__()
        self.num_channel = num_channel
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel)
        self.conv1 = nn.Conv1d(
            in_channels=self.num_channel,
            out_channels=self.num_channel * 2,
            kernel_size=(5,),
            stride=(1,),
            padding_mode="replicate",
            padding=4,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.num_channel * 2,
            out_channels=self.num_channel * 4,
            kernel_size=(5,),
            stride=(1,),
            padding_mode="replicate",
            padding=4,
        )
        self.conv3 = nn.Conv1d(
            in_channels=self.num_channel * 4,
            out_channels=self.num_channel * 8,
            kernel_size=(5,),
            stride=(1,),
            padding_mode="replicate",
            padding=4,
        )
        self.conv4 = nn.Conv1d(
            in_channels=self.num_channel * 8,
            out_channels=self.num_channel * 16,
            kernel_size=(5,),
            stride=(4,),
            padding_mode="replicate",
            padding=1,
        )

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, kernel_size=3)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(CNN_LSTM, self).__init__()
        self.CNN_ = CNN_(num_channel=num_channel)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=192, hidden_size=192 * 2, num_layers=lstm_num, batch_first=True
        )
        self.linear3 = nn.Linear(in_features=192 * 2 * 468, out_features=3)

    def forward(self, x, hidden=None):
        x = self.CNN_.forward(x)
        x = x.reshape(-1, 468, 192)
        x, hidden = self.lstm(x, hidden)
        # print(x.shape)
        x = x.reshape(-1, 468 * 384)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1


class N_VCNN(nn.Module):
    def __init__(self, num_channel, dropout):
        super(N_VCNN, self).__init__()
        self.num_channel = num_channel

        self.conv1 = nn.Conv1d(
            in_channels=self.num_channel,
            out_channels=self.num_channel * 16,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv2 = nn.Conv1d(
            in_channels=self.num_channel * 16,
            out_channels=self.num_channel * 8,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.num_channel * 8)
        self.conv3 = nn.Conv1d(
            in_channels=self.num_channel * 8,
            out_channels=self.num_channel * 4,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.bn3 = nn.BatchNorm1d(num_features=self.num_channel * 4)
        self.conv4 = nn.Conv1d(
            in_channels=self.num_channel * 4,
            out_channels=self.num_channel * 2,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.bn4 = nn.BatchNorm1d(num_features=self.num_channel * 2)
        self.conv5 = nn.Conv1d(
            in_channels=self.num_channel * 2,
            out_channels=self.num_channel,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_features=self.num_channel * 7500, out_features=3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu((self.conv5(x)))
        # print(x.shape)
        x = x.view(-1, 12*7500)
        x = self.linear1(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class N_VCNN_(nn.Module):
    def __init__(self, num_channel, dropout):
        super(N_VCNN_, self).__init__()
        self.num_channel = num_channel
        self.conv1 = nn.Conv1d(
            in_channels=self.num_channel,
            out_channels=self.num_channel * 16,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv2 = nn.Conv1d(
            in_channels=self.num_channel * 16,
            out_channels=self.num_channel * 8,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.num_channel * 8)
        self.conv3 = nn.Conv1d(
            in_channels=self.num_channel * 8,
            out_channels=self.num_channel * 4,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.bn3 = nn.BatchNorm1d(num_features=self.num_channel * 4)
        self.conv4 = nn.Conv1d(
            in_channels=self.num_channel * 4,
            out_channels=self.num_channel * 2,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.bn4 = nn.BatchNorm1d(num_features=self.num_channel * 2)
        self.conv5 = nn.Conv1d(
            in_channels=self.num_channel * 2,
            out_channels=self.num_channel,
            kernel_size=(3,),
            stride=(1,),
            padding_mode="replicate",
            padding=1,
        )
        self.linear1 = nn.Linear(in_features=7500*12,out_features=512)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu((self.conv5(x)))
        # print(x.shape)
        return x


class N_VCNN_LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(N_VCNN_LSTM, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2

        self.N_VCNN = N_VCNN_(num_channel, dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=self.num_channel,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_num,
        )
        self.linear3 = nn.Linear(
            in_features=self.lstm_hidden_size * 7500, out_features=512
        )

    def forward(self, x, hidden=None):
        x = self.N_VCNN.forward(x)
        x = x.reshape(-1, 7500, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        x = x.reshape(-1, self.lstm_hidden_size * 7500)
        x = self.dropout(x)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class N_VCNN_GRU(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(N_VCNN_GRU, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2
        self.N_VCNN = N_VCNN_(num_channel=num_channel, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

        self.gru = nn.GRU(
            input_size=self.num_channel,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_num,
        )
        self.linear3 = nn.Linear(
            in_features=self.lstm_hidden_size * 160, out_features=512
        )

    def forward(self, x, hidden=None):
        x = self.N_VCNN.forward(x)
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.gru(x, hidden)
        # print(x.shape)
        x = x.view(-1, self.lstm_hidden_size * 160)
        x = self.dropout(x)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class GRU(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num=2):
        super(GRU, self).__init__()
        self.num_channel = num_channel

        self.gru = nn.GRU(
            input_size=self.num_channel,
            hidden_size=self.num_channel * 2,
            dropout=dropout,
            num_layers=lstm_num,
        )
        self.linear = nn.Linear(
            in_features=self.num_channel * 2 * 160, out_features=512
        )

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.gru(x, hidden)
        x = x.view(-1, self.num_channel * 2 * 160)
        x = self.linear(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num=2):
        super(LSTM, self).__init__()
        self.num_channel = num_channel

        self.lstm = nn.LSTM(
            input_size=self.num_channel,
            hidden_size=self.num_channel * 2,
            dropout=dropout,
            num_layers=lstm_num,
        )
        self.linear = nn.Linear(
            in_features=self.num_channel * 2 * 160, out_features=512
        )

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        print(x.shape)
        x = x.view(-1, self.num_channel * 2 * 160)
        x = self.linear(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class LSTM_with_Attention(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num=2):
        super(LSTM_with_Attention, self).__init__()
        self.num_channel = num_channel
        self.lstm = nn.LSTM(
            input_size=self.num_channel,
            hidden_size=self.num_channel,
            dropout=dropout,
            num_layers=lstm_num,
        )
        self.linear = nn.Linear(in_features=self.num_channel * 160, out_features=512)
        self.attention_weight = nn.Parameter(
            torch.randn(self.num_channel * 160, self.num_channel * 160)
        )

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        x = x.view(-1, self.num_channel * 160)  # N,20480
        attention = F.softmax(torch.mm(x, self.attention_weight), dim=1)
        # print(attention.shape)  # N，self.num_channel * 2 * 160  ## N, 20480
        x = torch.mul(x, attention)  # N,20480
        x = self.linear(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class F_LSTM_CNN(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(F_LSTM_CNN, self).__init__()
        self.dropout = dropout
        self.num_channel = num_channel
        self.hidden_size = self.num_channel * 2
        self.CNN = CNN(num_channel=self.hidden_size, dropout=dropout)
        self.lstm1 = nn.LSTM(
            input_size=self.num_channel,
            hidden_size=self.hidden_size,
            num_layers=lstm_num,
        )

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm1(x, hidden)
        x = x.view(-1, self.hidden_size, 160)
        x = self.CNN(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


def main():
    net = LSTM_with_Attention(64, 0.2, 2)
    inputs = torch.randn(3, 64, 160)
    # print(inputs.shape)
    # results = net(inputs)
    results, _ = net(inputs)
    print(results.shape)

if __name__ == "__main__":
    main()
