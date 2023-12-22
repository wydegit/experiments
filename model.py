import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.fcn import FCNHead
from module import *
from combine import PCMLayer



class ALCNet_straight(nn.Module):
    """
     Resnet backbone + MPCM + BLAM + FCNHead
    """
    def __init__(self, backbone, same_layer=PCMLayer, cross_layer=BLAM, classes=1):
        super(ALCNet_straight, self).__init__()

        self.backbone = backbone  # tiny
        self.same_layer = same_layer(mpcm=True)

        channels = backbone.channels
        self.dec3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )  # 64 -> 16
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )  # 32 -> 16

        self.cross_layer = cross_layer(channels=channels[1], r=4)

        self.head = FCNHead(in_channels=channels[1], channels=classes)

    def forward(self, x):
        _, _, orig_h, orig_w = x.shape  # 480

        c1, c2, c3 = self.backbone(x)
        _, _, c1_h, c1_w = c1.shape  # 16 *480/120
        _, _, c2_h, c2_w = c2.shape  # 32 *240/60
        _, _, c3_h, c3_w = c3.shape  # 64 *120/30

        c3pcm = self.same_layer(c3)
        c3pcm = self.dec3(c3pcm)  # 64->16
        up_c3pcm = F.interpolate(c3pcm, size=(c2_h, c2_w), mode='bilinear')

        c2pcm = self.same_layer(c2)
        c2pcm = self.dec2(c2pcm)  # 32->16

        c23pcm = self.cross_layer(c2pcm, up_c3pcm)
        up_c23pcm = F.interpolate(c23pcm, size=(c1_h, c1_w), mode='bilinear')

        c1pcm = self.same_layer(c1)
        c123pcm = self.cross_layer(c1pcm, up_c23pcm)

        pred = self.head(c123pcm)
        out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear')

        return out


class ALCNet_cross(nn.Module):
    def __init__(self, backbone, same_layer=PCMLayer, cross_layer=BLAM, classes=1):
        super(ALCNet_cross, self).__init__()

        self.backbone = backbone  # tiny
        self.same_layer = same_layer(mpcm=True)

        channels = backbone.channels
        self.dec3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[2], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )  # 64 -> 32
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )  # 32 -> 16

        self.cross_layer1 = cross_layer(channels=channels[1], r=4)  # 16
        self.cross_layer2 = cross_layer(channels=channels[2], r=4)  # 32

        self.head = FCNHead(in_channels=channels[1], channels=classes)


    def forward(self, x):
        _, _, orig_h, orig_w = x.shape  # 480

        c1, c2, c3 = self.backbone(x)
        _, _, c1_h, c1_w = c1.shape  # 16 *480/120
        _, _, c2_h, c2_w = c2.shape  # 32 *240/60
        _, _, c3_h, c3_w = c3.shape  # 64 *120/30

        c3pcm = self.same_layer(c3)
        c2pcm = self.same_layer(c2)
        c1pcm = self.same_layer(c1)

        c3pcm = self.dec3(c3pcm)  # 64->32
        up_c3pcm = F.interpolate(c3pcm, size=(c2_h, c2_w), mode='bilinear')
        c23pcm = self.cross_layer2(c2pcm, up_c3pcm)  # 32

        c23pcm = self.dec2(c23pcm)  # 32->16
        up_c23pcm = F.interpolate(c23pcm, size=(c1_h, c1_w), mode='bilinear')
        c123pcm = self.cross_layer1(c1pcm, up_c23pcm)  # 16

        pred = self.head(c123pcm)
        out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear')

        return out





class FPN(nn.Module):
    """
    Resnet backbone + 0 + add + FCNHead (first change channels and upsample to add then predict)
    """
    def __init__(self, backbone, same_layer=None, cross_layer=add, fuse_direction='Dec', classes=1):
        super(FPN, self).__init__()
        self.backbone = backbone
        self.same_layer = same_layer
        self.cross_layer = cross_layer(16)

        layers = backbone.layers
        layers_num = list(range(1, len(layers) + 1))  # default [1,2,3]
        channels = backbone.channels  # defalut [8, 16, 32, 64]

        ## fuse_direction: increase('Inc'), decrease('Dec')
        ## fuse_process: 1*1 conv+bn+relu to equalize channels num
        if fuse_direction == 'Dec':  # outchannels->first(16)
            self.conv1 = nn.Conv2d(channels[layers_num[1]], channels[layers_num[0]], kernel_size=1, stride=1,
                                   padding=0, bias=False)  # 32->16
            self.bn1 = nn.BatchNorm2d(channels[layers_num[0]])

            self.conv2 = nn.Conv2d(channels[layers_num[2]], channels[layers_num[0]], kernel_size=1, stride=1,
                                   padding=0, bias=False)  # 64->16
            self.bn2 = nn.BatchNorm2d(channels[layers_num[0]])

            self.relu = nn.ReLU(inplace=True)

        elif fuse_direction == 'Inc': # outchannels->last(64)
            self.conv1 = nn.Conv2d(channels[layers_num[-2]], channels[layers_num[-1]], kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[layers_num[-1]])

            self.conv2 = nn.Conv2d(channels[layers_num[-3]], channels[layers_num[-1]], kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(channels[layers_num[-1]])

            self.relu = nn.ReLU(inplace=True)
        else:
            raise ValueError('Wrong direction!')

        self.head = FCNHead(channels[1], channels=classes)

    def forward(self, x):
        _, _, orig_h, orig_w = x.shape  # 512*512*1

        c1, c2, c3 = self.backbone(x)
        _, _, c1_h, c1_w = c1.shape  # 480*480*16
        _, _, c2_h, c2_w = c2.shape  # 240*240*32
        _, _, c3_h, c3_w = c3.shape  # 120*120*64

        c3 = self.relu(self.bn2(self.conv2(c3)))  #
        up_c3 = F.interpolate(c3, size=(c2_h, c2_w), mode='bilinear') # 240*240*16

        c2 = self.relu(self.bn1(self.conv1(c2)))  # 32->16

        c23 = self.cross_layer(up_c3, c2)
        up_c23 = F.interpolate(c23, size=(c1_h, c1_w), mode='bilinear')

        out = self.cross_layer(up_c23, c1)  # ->16

        pred = self.head(out)
        out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear')

        return out



class FCN(nn.Module):
    """
    Normal: Resnet backbone + 0 + 0 + FCNHead
    Add: Resnet backbone + 0 + FCNHead + add (first predict then upsample to add)
    """
    def __init__(self, backbone, same_layer=None, cross_layer=add, classes=1, fuse=False):
        super(FCN, self).__init__()

        self.backbone = backbone
        self.same_layer = same_layer
        self.cross_layer = cross_layer(1)
        self.fuse = fuse

        layers = backbone.layers
        layers_num = list(range(1, len(layers) + 1))  # default [1,2,3]
        channels = backbone.channels  # defalut [8, 16, 32, 64]


        self.head1 = FCNHead(channels[-3], channels=classes)  # 16->1
        self.head2 = FCNHead(channels[-2], channels=classes)  # 32->1
        self.head3 = FCNHead(channels[-1], channels=classes)  # 64->1


    def forward(self, x):

        _, _, orig_h, orig_w = x.shape  # 512*512*1

        c1, c2, c3 = self.backbone(x)
        _, _, c1_h, c1_w = c1.shape  # 480*480*16
        _, _, c2_h, c2_w = c2.shape  # 240*240*32
        _, _, c3_h, c3_w = c3.shape  # 120*120*64

        if self.fuse:
            # respectively predict-> add -> upsample
            c3_up = F.interpolate(c3, size=(c2_h, c2_w), mode='bilinear')  # 120*120*64->240*240*64
            c3_predict = self.head3(c3_up)  # 240*240*64->240*240*1

            c2_predict = self.head2(c2)  # 240*240*32->240*240*1

            c23 = self.cross_layer(c2_predict, c3_predict) # 240*240*1
            c23_up = F.interpolate(c23, size=(c1_h, c1_w), mode='bilinear') # 240*240*1->480*480*1

            c1_predict = self.head1(c1)  # 480*480*16->480*480*1

            out = self.cross_layer(c23_up, c1_predict)
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear')

        else:
            pred = self.head3(c3)  #
            out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear')

        return out



class UNet(nn.Module):
    """
    modified UNet with Resnet backbone + 0 + concatenation + FCNHead
    """
    def __init__(self, backbone, same_layer=None, cross_layer=concat, classes=1):
        super(UNet, self).__init__()

        self.backbone = backbone
        self.same_layer = same_layer

        layers = backbone.layers
        layers_num = list(range(1, len(layers) + 1))  # default [1,2,3]
        channels = backbone.channels  # defalut [8, 16, 32, 64]

        self.cross_layer1 = cross_layer(channels[layers_num[2]], channels[layers_num[1]])
        self.cross_layer2 = cross_layer(channels[layers_num[1]], channels[layers_num[0]])


        ## ?? 使1*1卷积变通道还是3*3卷积变通道好 # alc里写的是1*1
        self.conv1 = nn.Conv2d(channels[layers_num[2]], channels[layers_num[1]], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[layers_num[1]])
        self.conv2 = nn.Conv2d(channels[layers_num[1]], channels[layers_num[0]], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[layers_num[0]])
        self.relu = nn.ReLU(inplace=True)

        self.head = FCNHead(channels[1], channels=classes)

    def forward(self, x):
        _, _, orig_h, orig_w = x.shape  # 512*512*1

        c1, c2, c3 = self.backbone(x)
        _, _, c1_h, c1_w = c1.shape  # 480*480*16
        _, _, c2_h, c2_w = c2.shape  # 240*240*32
        _, _, c3_h, c3_w = c3.shape  # 120*120*64

        c3_up = F.interpolate(c3, size=(c2_h, c2_w), mode='bilinear')  #240*240*64
        c3_up = self.relu(self.bn1(self.conv1(c3_up)))  # 240*240*32

        c23_cat = self.cross_layer1(c3_up, c2)  # 240*240*32

        c23_up = F.interpolate(c23_cat, size=(c1_h, c1_w), mode='bilinear') # 480*480*32
        c23_up = self.relu(self.bn2(self.conv2(c23_up)))  # 480*480*16

        c123_cat = self.cross_layer2(c23_up, c1) # 480*480*16

        pred = self.head(c123_cat)
        out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear')

        return out



class ACMFPN_straight(nn.Module):
    """
    resnet20 + 0 + ACM + FCNHead (based on FPN)
    """
    def __init__(self, backbone, same_layer=None, cross_layer=ACM, classes=1):
        super(ACMFPN_straight, self).__init__()
        self.backbone = backbone
        self.same_layer = same_layer

        self.cross_layer = cross_layer(channels=16)

        channels = backbone.channels
        self.dec3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )  # 64->16
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )  # 32->16

        self.head = FCNHead(channels[1], channels=classes)

    def forward(self, x):
        _, _, orig_h, orig_w = x.shape  # 480

        c1, c2, c3 = self.backbone(x)
        _, _, c1_h, c1_w = c1.shape  # 16 *480/120
        _, _, c2_h, c2_w = c2.shape  # 32 *240/60
        _, _, c3_h, c3_w = c3.shape  # 64 *120/30

        c3 = self.dec3(c3)  # ->16
        c3_up = F.interpolate(c3, size=(c2_h, c2_w), mode='bilinear')

        c2 = self.dec2(c2)  # ->16
        c23 = self.cross_layer(c2, c3_up)
        c23_up = F.interpolate(c23, size=(c1_h, c1_w), mode='bilinear')

        c123 = self.cross_layer(c1, c23_up)

        pred = self.head(c123)
        out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear')

        return out



class ACMFPN_cross(nn.Module):
    """
    resnet20 + 0 + ACM + FCNHead (based on FPN)
    """
    def __init__(self, backbone, same_layer=None, cross_layer=ACM, classes=1):
        super(ACMFPN_cross, self).__init__()
        self.backbone = backbone
        self.same_layer = same_layer

        self.cross_layer1 = cross_layer(channels=16)
        self.cross_layer2 = cross_layer(channels=32)

        channels = backbone.channels
        self.dec_1 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )  # 32->16
        self.dec_2 = nn.Sequential(
            nn.Conv2d(channels[3], channels[2], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )  # 64->32

        self.head = FCNHead(channels[1], channels=classes)

    def forward(self, x):
        _, _, orig_h, orig_w = x.shape  # 480

        c1, c2, c3 = self.backbone(x)
        _, _, c1_h, c1_w = c1.shape  # 16 *480/120
        _, _, c2_h, c2_w = c2.shape  # 32 *240/60
        _, _, c3_h, c3_w = c3.shape  # 64 *120/30


        c3_up = F.interpolate(c3, size=(c2_h, c2_w), mode='bilinear')
        c3_up = self.dec_2(c3_up)  # ->32
        c23 = self.cross_layer2(c2, c3_up)

        c23_up = F.interpolate(c23, size=(c1_h, c1_w), mode='bilinear')
        c23_up = self.dec_1(c23_up)  # ->16
        out = self.cross_layer1(c1, c23_up)

        pred = self.head(out)
        out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear')

        return out








