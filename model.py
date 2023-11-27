import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.resnet import BasicBlock


class MPCMResNetFPN(nn.Module):
    def __init__(self, layers, channels, shift=3, classes=1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(MPCMResNetFPN, self).__init__()
        self.layer_num = len(layers)

        self.shift = shift

        stem_width = int(channels[0])
        self.stem = nn.Sequential()     # prefix="stem"
        self.stem.add_module("bn0", norm_layer(stem_width, affine=False))
        self.stem.add_module("conv1", nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False))
        self.stem.add_module("bn1", norm_layer(stem_width))
        self.stem.add_module("relu1", nn.ReLU())

        self.stem.add_module("conv2", nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False))
        self.stem.add_module("bn2", norm_layer(stem_width))
        self.stem.add_module("relu2", nn.ReLU())

        self.stem.add_module("conv3", nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False))
        self.stem.add_module("bn3", norm_layer(stem_width*2))
        self.stem.add_module("relu3", nn.ReLU())

        self.head = FCNHead(in_channels=channels[1], channels=classes)  # look

        self.layer1 = self._make_layer(block=BasicBlock, layers=layers[0], channels=channels[1], stride=1,
                                       stage_index=1, in_channels=channels[1])

        self.layer2 = self._make_layer(block=BasicBlock, layers=layers[1], channels=channels[2], stride=2,
                                       stage_index=2, in_channels=channels[1])

        self.layer3 = self._make_layer(block=BasicBlock, layers=layers[2], channels=channels[3], stride=2,
                                       stage_index=3, in_channels=channels[2])

        self.inc_c2 = nn.Sequential()
        self.inc_c2.add_module("inc_c2_conv", nn.Conv2d(channels[1], channels[3], kernel_size=1, stride=1,
                                                        padding=0, bias=False))
        self.inc_c2.add_module("inc_c2_bn", norm_layer(channels[-1]))
        self.inc_c2.add_module("inc_c2_relu", nn.ReLU())

        self.inc_c1 = nn.Sequential()
        self.inc_c1.add_module("inc_c1_conv", nn.Conv2d(channels[1], channels[3], kernel_size=1, stride=1,
                                                        padding=0, bias=False))
        self.inc_c1.add_module("inc_c1_bn", norm_layer(channels[-1]))
        self.inc_c1.add_module("inc_c1_relu", nn.ReLU())

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, norm_layer=nn.BatchNorm2d,
                    norm_kwargs=None):
        layer = nn.Sequential()     # prefix="layer{}".format(stage_index)
        downsample = (channels != in_channels) or (stride != 1)  # channels有变化或者stride不为1时，需要downsample
        layer.add_module("stage{}".format(stage_index),
                         block(channels, stride, downsample, in_channels=in_channels, norm_layer=norm_layer,
                               norm_kwargs=norm_kwargs, prefix=''))
        for _ in range(layers-1):
            layer.add_module("stage{}".format(stage_index),
                             block(channels, 1, False, in_channels=channels, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, prefix=''))
        return layer

    def forward(self, x):
        _, _, orig_h, orig_w = x.shape
        x = self.stem(x)

        c1 = self.layer1(x)

        _, _, c1_h, c1_w = c1.shape
        c2 = self.layer2(c1)

        _, _, c2_h, c2_w = c2.shape
        c3 = self.layer3(c2)

        _, _, c3_h, c3_w = c3.shape

        c3pcm = self.cal_pcm(c3, shift=self.shift)
        up_c3pcm = F.interpolate(c3pcm, size=(c2_h, c2_w), mode='bilinear', align_corners=False)

        inc_c2 = self.inc_c2(c2)
        c2pcm = self.cal_pcm(c2, shift=self.shift)

        c23pcm = up_c3pcm + c2pcm

        up_c23pcm = F.interpolate(c23pcm, size=(c1_h, c1_w), mode='bilinear', align_corners=False)

        inc_c1 = self.inc_c1(c1)
        c1pcm = self.cal_pcm(c1, shift=self.shift)

        out = up_c23pcm + c1pcm
        pred = self.head(out)
        out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return out


    def circ_shift(self, cen, shift):
        """
        cyclic shift compute for MPCM
        take each shift part and concatenate them according to the direction
        :param cen: center feature map  b*c*h*w
        :param shift: shift size
        :return:
        """

        _, _, h, w = cen.shape

        ###### B1 ######
        # (-d,-d) 左对角线
        B1_NW = cen[:, :, shift:, shift:]  # B1_NW is cen's SE
        B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
        B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
        B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
        B1_N = torch.cat([B1_NW, B1_NE], dim=3)
        B1_S = torch.cat([B1_SW, B1_SE], dim=3)
        B1 = torch.cat([B1_N, B1_S], dim=2)

        ###### B2 ######
        # (-d,0) 垂直
        B2_N = cen[:, :, shift:, :]  # B2_N is cen's S
        B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
        B2 = torch.cat([B2_N, B2_S], dim=2)

        ###### B3 ######
        # (-d,d) 右对角线
        B3_NW = cen[:, :, shift:, w - shift:]  # B3_NW is cen's SE
        B3_NE = cen[:, :, shift:, :w-shift]      # B3_NE is cen's SW
        B3_SW = cen[:, :, :shift, w-shift:]      # B3_SW is cen's NE
        B3_SE = cen[:, :, :shift, :w-shift]          # B1_SE is cen's NW
        B3_N = torch.cat([B3_NW, B3_NE], dim=3)
        B3_S = torch.cat([B3_SW, B3_SE], dim=3)
        B3 = torch.cat([B3_N, B3_S], dim=2)

        ###### B4 ######
        # (0,d) 水平
        B4_W = cen[:, :, :, w - shift:]  # B2_W is cen's E
        B4_E = cen[:, :, :, :w-shift]          # B2_E is cen's S
        B4 = torch.cat([B4_W, B4_E], dim=3)

        ##### B5 ######
        B5_NW = cen[:, :, h - shift:, w - shift:]  # B5_NW is cen's SE
        B5_NE = cen[:, :, h-shift:, :w-shift]      # B5_NE is cen's SW
        B5_SW = cen[:, :, :h-shift, w-shift:]      # B5_SW is cen's NE
        B5_SE = cen[:, :, :h-shift, :w-shift]          # B5_SE is cen's NW
        B5_N = torch.cat([B5_NW, B5_NE], dim=3)
        B5_S = torch.cat([B5_SW, B5_SE], dim=3)
        B5 = torch.cat([B5_N, B5_S], dim=2)

        ##### B6 ######
        B6_N = cen[:, :, h - shift:, :]  # B6_N is cen's S
        B6_S = cen[:, :, :h-shift, :]      # B6_S is cen's N
        B6 = torch.cat([B6_N, B6_S], dim=2)

        ##### B7 ######
        B7_NW = cen[:, :, h - shift:, shift:]  # B7_NW is cen's SE
        B7_NE = cen[:, :, h-shift:, :shift]      # B7_NE is cen's SW
        B7_SW = cen[:, :, :h-shift, shift:]      # B7_SW is cen's NE
        B7_SE = cen[:, :, :h-shift, :shift]          # B7_SE is cen's NW
        B7_N = torch.cat([B7_NW, B7_NE], dim=3)
        B7_S = torch.cat([B7_SW, B7_SE], dim=3)
        B7 = torch.cat([B7_N, B7_S], dim=2)

        ##### B8 ######
        B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
        B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
        B8 = torch.cat([B8_W, B8_E], dim=3)

        return B1, B2, B3, B4, B5, B6, B7, B8

    def cal_pcm(self, cen, shift):
        """
        MPCM Block
        :param cen: center feature map
        :param shift: shift size
        :return: MPCM map
        """
        B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(cen, shift=shift)
        s1 = (B1 - cen) * (B5 - cen)
        s2 = (B2 - cen) * (B6 - cen)
        s3 = (B3 - cen) * (B7 - cen)
        s4 = (B4 - cen) * (B8 - cen)   # transfer to tensor

        c12 = torch.minimum(s1, s2)
        c123 = torch.minimum(c12, s3)
        c1234 = torch.minimum(c123, s4)

        return c1234


# class ASKCResNetFPN(nn.Module):
#     def __init__(self, layers, channels, fuse_mode, act_dilation, classes=1, tinyFlag=False,
#                  norm_layer=nn.BatchNorm2d, norm_kwargs=None):
#         super(ASKCResNetFPN, self).__init__()
#
#         self.layer_num = len(layers)
#         self.tinyFlag = tinyFlag
#
#         stem_width = int(channels[0])
#
#         self.stem = nn.Sequential() # prefix="stem"
#         # 数据输入前先预处理一下
#         self.stem.add_module("bn0", norm_layer(stem_width, affine=False))
#
#         #
#         if tinyFlag:
#             self.stem.add_module("conv1", nn.Conv2d(3, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False))
#             self.stem.add_module("bn1", norm_layer(stem_width*2))
#             self.stem.add_module("relu1", nn.ReLU())
#
#         else:
#             self.stem.add_module("conv1", nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False))
#             self.stem.add_module("bn1", norm_layer(stem_width))
#             self.stem.add_module("relu1", nn.ReLU())
#
#             self.stem.add_module("conv2", nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False))
#             self.stem.add_module("bn2", norm_layer(stem_width))
#             self.stem.add_module("relu2", nn.ReLU())
#
#             self.stem.add_module("conv3", nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False))
#             self.stem.add_module("bn3", norm_layer(stem_width*2))
#             self.stem.add_module("relu3", nn.ReLU())
#             self.stem.add_module("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
#
#         # FCN的head
#         self.head =
#
#
#
#     def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, norm_layer=nn.BatchNorm2d,
#                     norm_kwargs=None):
#         layer = nn.Sequential()  # prefix="layer{}".format(stage_index)
#         downsample = (channels != in_channels) or (stride != 1)  # channels有变化或者stride不为1时，需要downsample
#         layer.add_module("block0", block(in_channels, channels, stride, downsample, norm_layer,
#                                           norm_kwargs=norm_kwargs, prefix=''))
#         for _ in range(layers-1):
#             layer.add_module(block(channels, stride=1, in_channels=channels, norm_layer=norm_layer, prefix='')
#
#         return layer
#
#     def _fuse_layer(self, fuse_mode, channels, act_dilation):
#         if fuse_mode == "Direct_Add"
#             fuse_layer =