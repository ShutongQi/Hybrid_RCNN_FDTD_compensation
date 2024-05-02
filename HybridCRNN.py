import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, INPUT_SIZE, len_dense):
        super(CRNN, self).__init__() 
        self.conv1 = nn.Conv2d(2, 8, kernel_size=7,stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5,stride=1)
        self.conv3 = nn.Conv2d(16,32,kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(32,64,kernel_size=3,stride=1)
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,stride=1)
        self.conv6 = nn.Conv2d(128,256,kernel_size=3,stride=1)
        self.conv7 = nn.Conv2d(256,512,kernel_size=3,stride=1)
        self.mp = nn.MaxPool2d(3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(10368,6000)
        self.fc2 = nn.Linear(6000,3000)
        self.fc3 = nn.Linear(3000,300)
        self.fc4 = nn.Linear(300, len_dense)
        
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=32,     
            num_layers=4,       
            batch_first=True,
        )
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, len_dense)
        
    def forward(self, f_point, pattern, time_step):
        in_size = pattern.size(0)

        pattern_out = self.relu(self.conv1(pattern))
        pattern_out = self.relu(self.conv2(pattern_out))
        pattern_out = self.relu(self.conv3(pattern_out))
        pattern_out = self.relu(self.conv4(pattern_out))
        pattern_out = self.relu(self.conv5(pattern_out))
        pattern_out = self.mp(pattern_out)
        pattern_out = self.mp(pattern_out)

        pattern_out = pattern_out.view(in_size, -1)
        pattern_out = self.relu(self.fc1(pattern_out))
        pattern_out = self.relu(self.fc2(pattern_out))
        pattern_out = self.relu(self.fc3(pattern_out))
        pattern_out = self.relu(self.fc4(pattern_out))
        pattern_out = pattern_out.unsqueeze(1)
        pattern_out = pattern_out.repeat((1,time_step,1))
        
        x = torch.cat((pattern_out,f_point),2)
        r_out, (h_n,h_c) = self.rnn(x, None)

        outs = [] 
        for ts in range(r_out.size(1)):   
            linear_out1 = self.relu(self.linear1(r_out[:, ts, :]))
            linear_out2 = self.relu(self.linear2(linear_out1))
            linear_out3 = self.relu(self.linear3(linear_out2))
            linear_out4 = self.linear4(linear_out3)
            out = self.out(linear_out4)
            outs.append(out)
            
        return torch.stack(outs, dim=1)
