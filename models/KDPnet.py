from __future__ import absolute_import

'''Convnet for DSprites. 
Luca Scimeca
'''
import torch.nn as nn
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['convnet']


class KDPNet(nn.Module):

    def __init__(self, num_classes=1000, no_of_channels=1, no_blocks=5):
        super(KDPNet, self).__init__()

        self.kernel_blocks = []
        for i in range(no_blocks):
            kernel_size = 1 + 2*i
            self.kernel_blocks += self._make_block(no_of_channels, kernel_size=(kernel_size, kernel_size))

        self.bn1 = nn.BatchNorm2d(16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2704, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        nn.Dropout2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, input_channels, kernel_size=(5, 5)):
        model = nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=kernel_size),
            nn.ReLU()
        )
        return model

    def forward(self, x):
        # Max pooling over a (2, 2) window
        mid_outputs = [kernel_block(x) for kernel_block in self.kernel_blocks]
        x = torch.cat(tuple(mid_outputs), -1) # todo check the concatenating dimension
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


def kdpnet(**kwargs):
    """
    Constructs a ConvNet model.
    """
    return KDPNet(**kwargs)