# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import torch.nn as nn
import torch
var_size = {
    'emg': {
        'in_size': 8,
        'ker_size': 9,
        'fc_size': 32*44
    },
    'stroke': {
        'in_size': 21,
        'ker_size': 9,
        'fc_size': 32*10
    }
}

rnn_param = {
    'stroke': {
        'in_size': 42,#42是因为噪声所以*2
        'hidden_size': 64,
        'num_layer': 1,
        'fc_size': 32*10
    },
    'stroke2d': {
        'in_size': 42,#42
        'hidden_size': 64,
        'num_layer': 1,
        'fc_size': 32*10
    },
    'stroke_onlyxyz': {
        'in_size': 4,#42是因为噪声所以*2
        'hidden_size': 64,
        'num_layer': 1,
        'fc_size': 32*10
    },
    'stroke_quickdraw': {
        'in_size': 4,#42是因为噪声所以*2
        'hidden_size': 64,
        'num_layer': 1,
        'fc_size': 32*10
    },

}

class ActNetwork(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=var_size[taskname]['in_size'], out_channels=16, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.in_features = var_size[taskname]['fc_size']

    def forward(self, x):
        # print(self.conv1(x).shape)# torch.Size([160, 16, 1, 28])
        x = self.conv2(self.conv1(x))
        # print(x.shape) torch.Size([160, 32, 1, 10])
        x = x.view(-1, self.in_features)
        # print(x.shape)#torch.Size([160, 320])
        return x

class RNN(nn.Module):
    def __init__(self, taskname):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(rnn_param[taskname]['in_size'], rnn_param[taskname]['hidden_size'], rnn_param[taskname]['num_layer'], batch_first=True)
        self.fc = nn.Linear(rnn_param[taskname]['hidden_size'], 320)
        self.in_features = rnn_param[taskname]['fc_size']
    def forward(self, x):
        #x(160,21,1,64)
        # x = torch.squeeze(x)
        x =x.transpose(1, 3).squeeze(2)
        #x(160,64,3)
        # print(x.shape)
        # 初始化隐藏状态
        # h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        # 前向传播 LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        # print(f'out:{out.shape}')
        # 将LSTM的输出传递到全连接层
        out = self.fc(out)

        return out