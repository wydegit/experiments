import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

class Backbone(nn.Module):
    def __init__(self, layers, channels, tiny=True):
        super(Backbone, self).__init__()

        stem_width = channels[0]
        if tiny:
            # 1->16
            self.stem = nn.Sequential(
                nn.BatchNorm2d(3),
                nn.Conv2d(3, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width * 2),
                nn.ReLU(inplace=True),
            )
        else:
            # 1->8->16
            self.stem = nn.Sequential(
                nn.BatchNorm2d(3),
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.layer1 = self._make_layer(block=BasicBlock, block_num=layers[0], in_channels=channels[1],
                                       out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=BasicBlock, block_num=layers[1], in_channels=channels[1],
                                       out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=BasicBlock, block_num=layers[2], in_channels=channels[2],
                                       out_channels=channels[3], stride=2)

        self.channels = channels

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        if (in_channels != out_channels) or (stride != 1):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            downsample = None

        layer.append(block(inplanes=in_channels, planes=out_channels, stride=stride, downsample=downsample))

        for _ in range(block_num - 1):
            layer.append(block(inplanes=out_channels, planes=out_channels, stride=1, downsample=None))

        return nn.Sequential(*layer)


    def forward(self, x):
        i1 = self.stem(x)  # if tiny:down 1; else down 4  16
        s1 = self.layer1(i1)   # stage1   down1/4    16
        s2 = self.layer2(s1)   # stage2   down2/8    32
        s3 = self.layer3(s2)   # stage3   down4/16   64

        return s1, s2, s3




######## cross layer fusion ########
### first down/upsample then change channels
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
    post_process : avoid the aliasing effect caused by upsample and feature fusion
    """
    def __init__(self, channels=16, r=4, post_process=False):
        super(BLAM, self).__init__()
        inter_channels = int(channels // r)
        self.channels = channels
        self.post_process = post_process

        self.bottomup_local = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x, y):
        """
        :param x: represents low-level feature map
        :param y: residual: represents high-level feature map
        :return: x + L(x) * Y  (L(x) is local attention map)
        """
        assert x.shape == y.shape
        assert x.shape[1] == self.channels

        lx = self.bottomup_local(x)
        fuse = x + lx * y

        if self.post_process:
            fuse = self.post(fuse)

        return fuse


class BGAM(nn.Module):
    """
    Bottum Up Global Attention Module for feature map fusion
    channels: input channel num
    r: reduction ratio
    """
    def __init__(self, channels=64, r=2, post_process=False):
        super(BGAM, self).__init__()
        inter_channels = int(channels // r)
        self.channels = channels
        self.post_process = post_process

        self.bottomup_global = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        """
        :param x: represents low-level feature map
        :param y: residual: represents high-level feature map
        :return: x + g(x) * Y  (g(x) is global attention map)
        """
        assert x.shape == y.shape
        assert x.shape[1] == self.channels

        gx = self.bottomup_global(x)
        fuse = x + gx * y

        if self.post_process:
            fuse = self.post(fuse)

        return fuse


class TLAM(nn.Module):
    """
    Top Down Local Attention Module for feature map fusion
    """
    def __init__(self, channels, r=4, post_process=False):
        super(TLAM, self).__init__()
        inter_channels = int(channels // r)
        self.channels = channels
        self.post_process = post_process

        self.topdown_local = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        """
        :param x: represents low-level feature map
        :param y: represents high-level feature map
        :return: L(y) * x  (L(y) is local attention map)
        """
        assert x.shape == y.shape
        assert x.shape[1] == self.channels

        ly = self.topdown_local(y)
        fuse = ly * x + y

        if self.post_process:
            fuse = self.post(fuse)

        return fuse


class TGAM(nn.Module):
    """
    Top Down Global Attention Module for feature map fusion
    """
    def __init__(self, channels, r=4, post_process=False):
        super(TGAM, self).__init__()
        inter_channels = int(channels // r)
        self.channels = channels

        self.topdown_global = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        """
        :param x: represents low-level feature map
        :param y: represents high-level feature map
        :return: G(y) * x  (G(y) is global attention map)
        """
        assert x.shape == y.shape
        assert x.shape[1] == self.channels

        gy = self.topdown_global(y)
        fuse = gy * x + y

        if self.post_process:
            fuse = self.post(fuse)

        return fuse


class ACM(nn.Module):
    """
    Asymmetric Contextual Module
    """
    def __init__(self, channels, r=4, post_process=True):
        super(ACM, self).__init__()
        inter_channels = int(channels // r)
        self.channels = channels
        self.post_process = post_process

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

    def forward(self, x, y):
        """
        :param x: represents low-level feature map
        :param y: represents high-level feature map
        :return: G(y) * x + L(x) * y
        """
        assert x.shape == y.shape
        assert x.shape[1] == self.channels

        gy = self.topdown(y)
        lx = self.bottomup(x)
        fuse = 2 * x * gy + 2 * y * lx  # 2?

        if self.post_process:
            fuse = self.post(fuse)

        return fuse







