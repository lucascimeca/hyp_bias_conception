from __future__ import absolute_import

'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch.nn as nn
import math
import utils.mode_connectivity.curves as curves

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # first block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # second block
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(
                x)
        # add residual
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # first block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # second block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # third block
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # add residual
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, no_of_channels=1):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) / 6
        n = int(n)

        block = Bottleneck if depth >= 44 else BasicBlock

        self.depth = depth
        self.num_classes = num_classes

        self.inplanes = 16
        self.conv1 = nn.Conv2d(no_of_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # 3 times of downsampling
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256*block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # adding layers
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


# ------- curve stuff --------------------------

def conv3x3Curve(in_planes, out_planes, fix_points, stride=1):
    # 3x3 convolution with padding
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, fix_points=fix_points)


class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fix_points=None):
        super(BasicBlockCurve, self).__init__()
        self.conv1 = conv3x3Curve(inplanes, planes, fix_points, stride)
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3Curve(planes, planes, fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        residual = x
        # first block
        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)
        # second block
        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)

        if self.downsample is not None:
            residual = self.downsample(x, coeffs_t)
        # add residual
        out += residual
        out = self.relu(out)
        return out


class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, fix_points, stride=1, downsample=None):
        super(BottleneckCurve, self).__init__()
        self.conv1 = curves.Conv2d(inplanes, planes,
                                   kernel_size=1,
                                   bias=False,
                                   fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1,
                                   bias=False,
                                   fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv3 = curves.Conv2d(planes, planes * 4,
                                   kernel_size=1,
                                   bias=False,
                                   fix_points=fix_points)
        self.bn3 = curves.BatchNorm2d(planes * 4, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        residual = x
        # first block
        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)
        # second block
        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)
        out = self.relu(out)
        # third block
        out = self.conv3(out, coeffs_t)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x, coeffs_t)

        # add residual
        out += residual
        out = self.relu(out)
        return out




class DownSample(nn.Module):
    def __init__(self, inplanes, planes, expansion, fix_points=None, kernel_size=1, stride=1, bias=False):
        super(DownSample, self).__init__()
        self.conv = curves.Conv2d(inplanes, planes * expansion, fix_points=fix_points, kernel_size=kernel_size, stride=stride, bias=bias)
        self.batch_norm = curves.BatchNorm2d(planes * expansion, fix_points=fix_points)

    def forward(self, x, coeffs_t):
        x = self.conv(x, coeffs_t)
        x = self.batch_norm(x, coeffs_t)
        return x


class ResNetCurve(nn.Module):
    def __init__(self, depth, fix_points, num_classes=1000, no_of_channels=1):
        super(ResNetCurve, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) / 6
        n = int(n)

        block = BottleneckCurve if depth >= 44 else BasicBlockCurve

        self.inplanes = 16
        self.conv1 = curves.Conv2d(no_of_channels, 16, fix_points=fix_points, kernel_size=3, padding=1, bias=False)
        self.bn1 = curves.BatchNorm2d(16, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        # 3 times of downsampling
        self.layer1 = self._make_layer(block, 16, n, fix_points=fix_points)
        self.layer2 = self._make_layer(block, 32, n, fix_points=fix_points, stride=2)
        self.layer3 = self._make_layer(block, 64, n, fix_points=fix_points, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = curves.Linear(256*block.expansion, num_classes, fix_points=fix_points)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, fix_points, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownSample(self.inplanes, planes, block.expansion, fix_points=fix_points, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fix_points=fix_points))
        self.inplanes = planes * block.expansion
        # adding layers
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, fix_points=fix_points))

        return nn.Sequential(*layers)

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        x = self.bn1(x, coeffs_t)
        x = self.relu(x)

        for block in self.layer1:
            x = block(x, coeffs_t)  # 32x32
        for block in self.layer2:
            x = block(x, coeffs_t)  # 16x16
        for block in self.layer3:
            x = block(x, coeffs_t)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, coeffs_t)
        return x


class ResnetWithCurve:
    base = ResNet
    curve = ResNetCurve