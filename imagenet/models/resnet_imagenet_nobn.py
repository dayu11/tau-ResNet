import torch
import torch.nn as nn
import numpy as np
import math




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class Bottleneck_NoBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, tau, stride=1, downsample=None):
        super(Bottleneck_NoBN, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.tau = tau
 
        #self.bn1 = nn.BatchNorm2d(planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # Follow Fixup (Zhang et al. 2018), we add scalar terms before/after each convolution layer
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.bias3b = nn.Parameter(torch.zeros(1))        
        

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        #out = self.bn1(out)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        #out = self.bn2(out)
        out = self.relu(out + self.bias2b)

        out = (self.conv3(out + self.bias3a) + self.bias3b) * self.tau 
        #out = self.bn3(out) 
        

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_NoBN(nn.Module):

    def __init__(self, block, layers, tau, num_classes=1000):
        super(ResNet_NoBN, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.depth = self.num_layers * 3 + 2


        #self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.tau = nn.Parameter(torch.ones(1)*tau)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], self.tau)
        self.layer2 = self._make_layer(block, 128, layers[1], self.tau, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], self.tau, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], self.tau, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #standard initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, tau, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride))

        layers = []
        layers.append(block(self.inplanes, planes, tau, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, tau))

        return nn.Sequential(*layers)

    def scale_tau_grad(self):
        self.tau.grad /= self.num_layers

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet50_nobn(tau):
    """Constructs a Fixup-ResImagenet-50 model.

    """
    model = ResNet_NoBN(Bottleneck_NoBN, [3, 4, 6, 3], tau)
    return model


def resnet101_nobn(tau):
    """Constructs a Fixup-ResImagenet-101 model.

    """
    model = ResNet_NoBN(Bottleneck_NoBN, [3, 4, 23, 3], tau)
    return model


def resnet152_nobn(tau):
    """Constructs a Fixup-ResImagenet-152 model.

    """
    model = ResNet_NoBN(Bottleneck_NoBN, [3, 8, 36, 3], tau)
    return model

"""
cel = nn.CrossEntropyLoss()
#num_classes, dropout, tau, nobn, learn_tau, dspbn
inputs = torch.zeros(120, 3, 256, 256).cuda()
targets = torch.zeros((120, ), dtype=torch.long).cuda()
dummy_targets = torch.zeros(240, 1000).cuda()
net=ResImagenet(Bottleneck, [3, 8, 36, 3], 1000, 0., 0.1, 1, True, True)
net.cuda()
net = torch.nn.DataParallel(net)
while (True):
    out = net(inputs)
    loss = cel(out, targets)
    loss.backward()
"""