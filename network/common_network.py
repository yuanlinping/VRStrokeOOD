# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
import torch

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)#feature_dim = torch.Size([160, 32*44])
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
    
import torch
import torch.nn as nn

#虽然叫regressor但它现在变成了generator
class feat_regressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128]):
        super(feat_regressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.fc(x)
        return x
    
class cgan_discriminator(nn.Module):
    def __init__(self,input_dim=64,output_dim=1,hidden_dims=[256, 128],model_name = "CNN"):
        super(cgan_discriminator,self).__init__()
        self.model_name = model_name
        if model_name == "CNN":
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=(1, 3), stride=1, padding=(0, 1)),#input = 22
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Flatten(),
                nn.Linear(64 * 16, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, output_dim),
            )
        elif model_name == "LSTM":
            self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=1, batch_first=True)
            self.fc = nn.Linear(64, output_dim)
        else: 
            print("No such discriminator model")

    def forward(self, x):
        if self.model_name == "CNN":
            output = self.model(x)
        elif self.model_name == "LSTM":
            x = x.transpose(1, 3).squeeze(2)
            out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_size)
            out = out[:, -1, :]  # 取最后一个时间步的输出
            output = self.fc(out)#(batch_size,64)->(batch_size,2)
        else:
            print("No such discriminator model")
        return output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        x = x.unsqueeze(2)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        out = torch.squeeze(out)

        return out
