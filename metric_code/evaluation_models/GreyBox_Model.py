# Description: Definition of the grey box model architecture (same as white box)
# Author: Johannes Geier
# Date: 14.02.2025

import torch.nn as nn
import torch.nn.functional as F

class SubstituteNet(nn.Module):
    def __init__(self):
        super(SubstituteNet, self).__init__()
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )

        self.mlp_net = nn.Sequential(
            nn.Linear(8*8*128,128,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,128,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
        
    def forward(self, x):
        x = self.conv_net(x)
        x = self.mlp_net(x.flatten(start_dim=1))
        return F.softmax(x, dim = 1)