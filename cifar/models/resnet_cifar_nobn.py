import torch
import torch.nn as nn
import numpy as np

import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock_NoBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, tau, stride=1, downsample=None):
        super(BasicBlock_NoBN, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

        self.tau = tau
        # Follow Fixup (Zhang et al. 2018), we add scalar terms before/after each convolution layer
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        #out = self.bn1(out)
        out = self.relu(out + self.bias1b)

        out = (self.conv2(out + self.bias2a) + self.bias2b) * self.tau 
        #out = self.bn2(out) * self.tau

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class ResNet_NoBN(nn.Module):

    def __init__(self, block, layers, tau, num_classes=10):
        super(ResNet_NoBN, self).__init__()
        
        self.tau = nn.Parameter(torch.ones(1)*tau) # all blocks share the same learnable tau

        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        #self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], self.tau)
        self.layer2 = self._make_layer(block, 32, layers[1], self.tau, stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], self.tau, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        # standard initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
    def _make_layer(self, block, planes, blocks, tau, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                #nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, tau, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, tau))

        return nn.Sequential(*layers)

    def scale_tau_grad(self): # take the average gradient from all blocks
        self.tau.grad /= self.num_layers

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_nobn(tau):
    """Constructs a ResNet-20 model.

    """
    model = ResNet_NoBN(BasicBlock_NoBN, [3, 3, 3], tau)
    return model


def resnet32_nobn(tau):
    """Constructs a ResNet-32 model.

    """
    model = ResNet_NoBN(BasicBlock_NoBN, [5, 5, 5], tau)
    return model


def resnet44_nobn(tau):
    """Constructs a ResNet-44 model.

    """
    model = ResNet_NoBN(BasicBlock_NoBN, [7, 7, 7], tau)
    return model


def resnet56_nobn(tau):
    """Constructs a ResNet-56 model.

    """
    model = ResNet_NoBN(BasicBlock_NoBN, [9, 9, 9], tau)
    return model


def resnet110_nobn(tau):
    """Constructs a ResNet-110 model.

    """
    model = ResNet_NoBN(BasicBlock_NoBN, [18, 18, 18], tau)
    return model


def resnet1202_nobn(tau):
    """Constructs a ResNet-1202 model.

    """
    model = ResNet_NoBN(BasicBlock_NoBN, [200, 200, 200], tau)
    return model    