import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, downsample):
        super(Bottleneck, self).__init__()
        channels = in_channels // 2

        self.downsample = downsample
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(channels, in_channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu(out)

        out += x  # Residual
        return out


class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()
        self.in_channels = 32
        self.conv1 = conv3x3(3, self.in_channels)  # Input is an RGB image
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 8)
        self.layer4 = self._make_layer(512, 8)
        self.layer5 = self._make_layer(1024, 4)

    def _make_layer(self, channels, blocks):
        layers = []
        for i in range(blocks):
            downsample = None
            if self.in_channels != channels:
                downsample = nn.Sequential(
                    conv3x3(self.in_channels, channels, stride=2),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                self.in_channels = channels

            layers.append(Bottleneck(channels, downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out1 = self.layer3(x)
        out2 = self.layer4(out1)
        out3 = self.layer5(out2)

        return out1, out2, out3
