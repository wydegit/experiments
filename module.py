import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

class Backbone(nn.Module):
    """
    defalut Resnet backbone
    layers = [args.blocks] * stage_num , the num of basickblock/bottleneck in each stage
    stage_num  default:3
    """
    def __init__(self, layers, channels):
        super(Backbone, self).__init__()

        # 1->16
        self.conv1 = nn.Conv2d(1, channels[0] * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0] * 2)
        self.relu = nn.ReLU(inplace=True)



        self.layer1 = self._make_layer(block=BasicBlock, layers=layers[0], channels=channels[1], stride=1,
                                       stage_index=1, in_channels=channels[1])

        self.layer2 = self._make_layer(block=BasicBlock, layers=layers[1], channels=channels[2], stride=2,
                                       stage_index=2, in_channels=channels[1])

        self.layer3 = self._make_layer(block=BasicBlock, layers=layers[2], channels=channels[3], stride=2,
                                       stage_index=3, in_channels=channels[2])

        self.layers = layers
        self.channels = channels

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels, norm_layer=nn.BatchNorm2d,
                    norm_kwargs=None):
        layer = nn.Sequential()     # prefix="layer{}".format(stage_index)

        if (channels != in_channels) or (stride != 1):  # channels有变化或者stride不为1时，需要downsample
            downsample = nn.Sequential()
            downsample.add_module("downsample_conv", nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride,
                                                               bias=False))
            downsample.add_module("downsample_bn", norm_layer(channels))
        else:
            downsample = None

        layer.add_module("stage{}_1".format(stage_index), block(inplanes=in_channels, planes=channels, stride=stride,
                                                                downsample=downsample, norm_layer=norm_layer))

        for i in range(layers-1):  # 0，1，2
            layer.add_module("stage{}_{}".format(stage_index, i+2), block(inplanes=channels, planes=channels, stride=1,
                                                                          downsample=None, norm_layer=norm_layer))

        return layer

    def forward(self, x):
        i1 = self.relu(self.bn1(self.conv1(x)))
        s1 = self.layer1(i1)   # stage1
        s2 = self.layer2(s1)   # stage2
        s3 = self.layer3(s2)   # stage3

        return s1, s2, s3





######## cross layer fusion ########
class add(nn.Module):
    """
    Pixwel-wise add two feature maps
    """
    def __init__(self, channels=16):
        super(add, self).__init__()

        self.bn0 = nn.BatchNorm2d(channels)

    def forward(self, x, y):
        # x = self.bn0(x)
        # y = self.bn0(y)

        fuse = x + y

        return fuse

class concat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(concat, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, y):

        cat = torch.cat((x, y), dim=1)
        fuse = self.relu(self.bn1(self.conv1(cat)))

        return fuse



class BLAM(nn.Module):
    """
    Bottum Up Local Attention Module for feature map fusion
    channels: input channel num
    r: reduction ratio
    """
    def __init__(self, channels=16, r=2):
        super(BLAM, self).__init__()
        inter_channels = int(channels // r)

        self.bn0 = nn.BatchNorm2d(channels)

        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        :param x: represents low-level feature map
        :param y: residual: represents high-level feature map
        :return: x + L(x) * Y  (L(x) is local attention map)
        """
        x = self.bn0(x)
        y = self.bn0(y)

        lx = self.relu(self.bn1(self.conv1(x)))
        lx = self.sigmoid(self.bn2(self.conv2(lx)))

        fuse = x + lx * y

        return fuse


class BGAM(nn.Module):
    """
    Bottum Up Global Attention Module for feature map fusion
    channels: input channel num
    r: reduction ratio
    """
    def __init__(self, channels=64, r=2):
        super(BGAM, self).__init__()
        inter_channels = int(channels // r)

        self.bn0 = nn.BatchNorm2d(channels)
        self.GloAvgPool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        :param x: represents low-level feature map
        :param y: residual: represents high-level feature map
        :return: x + g(x) * Y  (g(x) is global attention map)
        """
        x = self.bn0(x)
        y = self.bn0(y)

        gx = self.GloAvgPool(x)
        gx = self.relu(self.bn1(self.conv1(gx)))
        gx = self.sigmoid(self.bn2(self.conv2(gx)))

        fuse = x + gx * y

        return fuse




