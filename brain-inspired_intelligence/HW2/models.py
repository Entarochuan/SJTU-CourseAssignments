"""

    filename : models.py
    author : Yichuan Ma
    Date : 2022/11/12
    Description : 实现了相同结构的SNN , CNN模型
    Reference :  Spikingjelly Document
    
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing
import torch
import torch.nn as nn

class SNN_Net(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T

        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )
        
        self.net = nn.Sequential(
            nn.Conv2d(8, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 7 * 7
            
            nn.Flatten(),
            nn.Linear(2048, 10, bias=False, dtype=torch.float32),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
            )
        
    def forward(self, x):

        # x = self.static_conv(x)
        # print(x.shape)
        # print('0 col shape', x[:,0,:,:,:].shape)
        
        out_spikes_counter = self.net(self.conv(x[0]))
        for t in range(1, self.T):
            out_spikes_counter += self.net(self.conv(x[t]))

        return out_spikes_counter / self.T
    
class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)
    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)
    
class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(channels))
            conv.append(spiking_neuron(*args, **kwargs))
            conv.append(nn.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels * 4 * 4, 512),
            spiking_neuron(*args, **kwargs),

            nn.Dropout(0.5),
            nn.Linear(512, 110),
            spiking_neuron(*args, **kwargs),

            VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):

        out_spikes = self.conv_fc(x[0])

        for t in range(1, 8):
            out_spikes += self.conv_fc(x[t])
            
        return out_spikes / x.shape[0]

    
    
class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 10, bias=False, dtype=torch.float32),
            )

    def forward(self, x):

        x = self.static_conv(x)
        
        x = self.net(self.conv(x))

        return x 