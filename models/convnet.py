from __future__ import absolute_import

'''Convnet for DSprites. 
Luca Scimeca
'''
import torch.nn as nn
import math


__all__ = ['convnet']


import torch
import torch.nn as nn
import torch.nn.functional as F

KERNEL_SIZE = 5

class ConvNet(nn.Module):

    def __init__(self, num_classes=1000, no_of_channels=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(no_of_channels, 8, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=KERNEL_SIZE)
        self.bn1 = nn.BatchNorm2d(16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2704, 120)  # 6*6 from image dimension - 2704 3x3, 4096 1x1, 1296 10x10
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), 2), p=0.3)
        x = self.bn1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def convnet(**kwargs):
    """
    Constructs a ConvNet model.
    """
    return ConvNet(**kwargs)