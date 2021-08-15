from __future__ import absolute_import

'''Convnet for DSprites. 
Luca Scimeca
'''

import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['convnet']


class FFNet(nn.Module):

    def __init__(self, num_classes=1000, no_of_channels=1):
        super(FFNet, self).__init__()
        self.fc1 = nn.Linear(no_of_channels*4096, 2048)
        self.fc2 = nn.Linear(2048, 600)
        self.fc3 = nn.Linear(600, 300)
        self.fc4 = nn.Linear(300, 100)
        self.fc5 = nn.Linear(100, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def ffnet(**kwargs):
    """
    Constructs a ConvNet model.
    """
    return FFNet(**kwargs)


class FFNetTest(nn.Module):

    def __init__(self, num_classes=1000, no_of_channels=1, depth=4):
        super(FFNetTest, self).__init__()
        self.depth = depth
        self.in_layer = nn.Linear(no_of_channels*64*64, 10)
        self.mid_layer = nn.Linear(10, 10)
        self.out_layer = nn.Linear(10, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.in_layer(x))
        for i in range(self.depth):
            x = F.relu(self.mid_layer(x))
        x = self.out_layer(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def ffnettest(**kwargs):
    """
    Constructs a ConvNet model.
    """
    return FFNetTest(**kwargs)