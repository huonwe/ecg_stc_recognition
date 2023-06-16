from future_model import *

class CNN_CNN_LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(CNN_CNN_LSTM, self).__init__()
        self.CNN = CNN(num_channel=num_channel, dropout=dropout)
        self.CNN_LSTM = CNN_LSTM(
            num_channel=num_channel, dropout=dropout, lstm_num=lstm_num
        )

    def forward(self, x, hidden=None):
        x1 = self.CNN(x)
        x2, hidden = self.CNN_LSTM(x)
        x = x1 + x2
        return x, hidden
    
    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1

class CNN_VCNN(nn.Module):
    def __init__(self, num_channel, dropout):
        super(CNN_VCNN, self).__init__()
        self.CNN_ = CNN_(num_channel=num_channel)
        self.VCNN_ = N_VCNN_(
            num_channel=num_channel, dropout=dropout
        )
        self.linear1 = nn.Linear(in_features=192*23, out_features=128)
        self.linear2 = nn.Linear(in_features=12*7500, out_features=128)
        self.linear3 = nn.Linear(in_features=128,out_features=3)

    def forward(self, x):
        x1 = self.CNN_(x)
        # print(x1.shape)
        x1 = x1.reshape(-1, 192*23)
        x2 = self.VCNN_(x)
        x2 = x2.reshape(-1,12*7500)
        
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x = self.linear3(x1+x2)
        return x
    
    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1