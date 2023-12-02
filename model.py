import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.resnet import BasicBlock


class ALCNet(nn.Module):
    def __init__(self, layers, channels, shift=13, pyramid_mode='xxx', scale_mode='xxx',
                 pyramid_fuse='xxx', r=2, classes=1, norm_layer=nn.BatchNorm2d, norm_kwargs=None,):
        super(ALCNet, self).__init__()

        self.layer_num = len(layers)

        self.r = r
        self.shift = shift
        self.pyramid_mode = pyramid_mode
        self.scale_mode = scale_mode
        self.pyramid_fuse = pyramid_fuse

        stem_width = int(channels[0])
        self.stem = nn.Sequential()  # prefix="stem"
        self.stem.add_module("bn0", norm_layer(3, affine=False))   # 改inchannels
        self.stem.add_module("conv1", nn.Conv2d(3, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False))
        self.stem.add_module("bn1", norm_layer(stem_width * 2))
        self.stem.add_module("relu1", nn.ReLU())


        self.head = FCNHead(in_channels=channels[1], channels=classes)

        self.layer1 = self._make_layer(block=BasicBlock, layers=layers[0], channels=channels[1], stride=1,
                                       stage_index=1, in_channels=channels[1])

        self.layer2 = self._make_layer(block=BasicBlock, layers=layers[1], channels=channels[2], stride=2,
                                       stage_index=2, in_channels=channels[1])

        self.layer3 = self._make_layer(block=BasicBlock, layers=layers[2], channels=channels[3], stride=2,
                                       stage_index=3, in_channels=channels[2])


        if pyramid_mode == 'Dec':
            self.dec_c2 = nn.Sequential()
            self.dec_c2.add_module("dec_c2_conv", nn.Conv2d(channels[2], channels[1], kernel_size=1, stride=1,
                                                            padding=0, bias=False))   # 改inchannels
            self.dec_c2.add_module("dec_c2_bn", norm_layer(channels[1]))
            self.dec_c2.add_module("dec_c2_relu", nn.ReLU())

            self.dec_c3 = nn.Sequential()
            self.dec_c3.add_module("dec_c3_conv", nn.Conv2d(channels[3], channels[1], kernel_size=1, stride=1,
                                                            padding=0, bias=False))
            self.dec_c3.add_module("dec_c3_bn", norm_layer(channels[1]))
            self.dec_c3.add_module("dec_c3_relu", nn.ReLU())

            self.cm = PCMLayer(mpcm=True)

            if self.pyramid_fuse == 'bottomuplocal':
                self.bottomuplocal_fpn_2 = BottomUpLocal_FPNFuse(channels=channels[1])   # 改inchannels
                self.bottomuplocal_fpn_1 = BottomUpLocal_FPNFuse(channels=channels[1])
            elif self.pyramid_fuse == 'bottomupglobal':
                self.bottomupglobal_fpn_2 = BottomUpGlobal_FPNFuse(channels=channels[1])
                self.bottomupglobal_fpn_1 = BottomUpGlobal_FPNFuse(channels=channels[1])
            else:
                raise ValueError("unknown pyramid_fuse")


        elif pyramid_mode == 'Inc':
            self.inc_c2 = nn.Sequential()
            self.inc_c2.add_module("inc_c2_conv", nn.Conv2d(channels[2], channels[3], kernel_size=1, stride=1,
                                                            padding=0, bias=False))
            self.inc_c2.add_module("inc_c2_bn", norm_layer(channels[-1]))
            self.inc_c2.add_module("inc_c2_relu", nn.ReLU())

            self.inc_c1 = nn.Sequential()
            self.inc_c1.add_module("inc_c1_conv", nn.Conv2d(channels[1], channels[3], kernel_size=1, stride=1,
                                                            padding=0, bias=False))
            self.inc_c1.add_module("inc_c1_bn", norm_layer(channels[-1]))
            self.inc_c1.add_module("inc_c1_relu", nn.ReLU())

        else:
            raise ValueError("unknown pyramid_mode")


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

        _, _, orig_h, orig_w = x.shape
        x = self.stem(x)  # 480*480*16

        c1 = self.layer1(x)
        _, _, c1_h, c1_w = c1.shape  # 480*480*16

        c2 = self.layer2(c1)
        _, _, c2_h, c2_w = c2.shape  # 240*240*32

        c3 = self.layer3(c2)
        _, _, c3_h, c3_w = c3.shape  # 120*120*64

        # 1. upsampling(c3) -> c3PCM   # size: sub 4

        # c3 -> c3PCM
        # 2. pwconv(c2) -> c2PCM       # size: sub 4
        # 3. upsampling(c3PCM + c2PCM) # size: sub 2
        # 4. pwconv(c1) -> c1PCM       # size: sub 2
        # 5. upsampling(upsampling(c3PCM + c2PCM)) + c1PCM
        # 6. upsampling(upsampling(c3PCM + c2PCM)) + c1PCM

        if self.pyramid_mode == 'Dec':
            c3pcm = self.cm(c3)  # sub 8, 64
            c3pcm = self.dec_c3(c3pcm)                  # sub 8, 16
            up_c3pcm = F.interpolate(c3pcm, size=(c2_h, c2_w), mode='bilinear')

            c2pcm = self.cm(c2)  # sub 4, 32
            c2pcm = self.dec_c2(c2pcm)                  # sub 4, 16

            if self.pyramid_fuse == 'bottomuplocal':
                c23pcm = self.bottomuplocal_fpn_2(up_c3pcm, c2pcm)  # (Y,X)
            elif self.pyramid_fuse == 'bottomupglobal':
                c23pcm = self.bottomupglobal_fpn_2(up_c3pcm, c2pcm)
            else:
                raise ValueError("unknown pyramid_fuse")
            up_c23pcm = F.interpolate(c23pcm, size=(c1_h, c1_w), mode='bilinear')  # sub 2, 16

            c1pcm = self.cm(c1)  # sub 2, 16

            if self.pyramid_fuse == 'bottomuplocal':
                out = self.bottomuplocal_fpn_1(up_c23pcm, c1pcm)
            elif self.pyramid_fuse == 'bottomupglobal':
                out = self.bottomupglobal_fpn_1(up_c23pcm, c1pcm)
            else:
                raise ValueError("unknown self.pyramid_fuse")

        elif self.pyramid_mode == 'Inc':

            c3pcm = self.cal_pcm(c3, shift=self.shift)
            up_c3pcm = F.interpolate(c3pcm, size=(c2_h, c2_w), mode='bilinear', align_corners=False)

            inc_c2 = self.inc_c2(c2)  # same channal size as c3 64
            c2pcm = self.cal_pcm(c2, shift=self.shift)

            c23pcm = up_c3pcm + c2pcm

            up_c23pcm = F.interpolate(c23pcm, size=(c1_h, c1_w), mode='bilinear', align_corners=False)
            inc_c1 = self.inc_c1(c1)
            c1pcm = self.cal_pcm(c1, shift=self.shift)

            out = up_c23pcm + c1pcm
        else:
            raise ValueError("unknown pyramid_mode")

        # out size : 480*480*64
        pred = self.head(out)
        out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear')

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)



class BottomUpLocal_FPNFuse(nn.Module):
    def __init__(self, channels=64):  # channels=16
        super(BottomUpLocal_FPNFuse, self).__init__()
        inter_channels = int(channels // 2)

        self.bn = nn.BatchNorm2d(channels)

        self.bottomup_att = nn.Sequential()
        self.bottomup_att.add_module("bottomupL_att_conv1", nn.Conv2d(in_channels=channels, out_channels=inter_channels,
                                                                          kernel_size=1, stride=1, padding=0))
        self.bottomup_att.add_module("bottomupL_att_bn1", nn.BatchNorm2d(inter_channels))
        self.bottomup_att.add_module("bottomupL_att_relu1", nn.ReLU())
        self.bottomup_att.add_module("bottomupL_att_conv2", nn.Conv2d(in_channels=inter_channels, out_channels=channels,
                                                                            kernel_size=1, stride=1, padding=0))
        self.bottomup_att.add_module("bottomupL_att_bn2", nn.BatchNorm2d(channels))
        self.bottomup_att.add_module("bottomupL_att_sigmoid", nn.Sigmoid())


    def forward(self, x, residual):  # Y，X

        x = self.bn(x)
        residual = self.bn(residual)

        bottomup_wei = self.bottomup_att(residual)
        xo = bottomup_wei * x + residual

        return xo


class BottomUpGlobal_FPNFuse(nn.Module):
    def __init__(self, channels=64):
        super(BottomUpGlobal_FPNFuse, self).__init__()
        inter_channels = int(channels // 1)

        self.bn = nn.BatchNorm2d(channels)

        self.bottomup_att = nn.Sequential()
        self.bottomup_att.add_module('botomupG_GloAvgPool', nn.AdaptiveAvgPool2d(1))
        self.bottomup_att.add_module("bottomupG_att_conv1", nn.Conv2d(in_channels=channels, out_channels=inter_channels,
                                                                      kernel_size=1, stride=1, padding=0))
        self.bottomup_att.add_module("bottomupL_att_bn1", nn.BatchNorm2d(inter_channels))
        self.bottomup_att.add_module("bottomupL_att_relu1", nn.ReLU())
        self.bottomup_att.add_module("bottomupL_att_conv2", nn.Conv2d(in_channels=inter_channels, out_channels=channels,
                                                                      kernel_size=1, stride=1, padding=0))
        self.bottomup_att.add_module("bottomupL_att_bn2", nn.BatchNorm2d(channels))
        self.bottomup_att.add_module("bottomupL_att_sigmoid", nn.Sigmoid())


    def forward(self, x, residual):

        x = self.bn(x)
        residual = self.bn(residual)

        bottomup_wei = self.bottomup_att(residual)

        xo = bottomup_wei * x + residual

        return xo





class PCMLayer(nn.Module):
    def __init__(self, mpcm=False):
        super(PCMLayer, self).__init__()
        self.mpcm = mpcm

    def __call__(self, cen):    # 改成 shift分别传入取最大
        self.cen = cen
        if self.mpcm:
            # MLC   :squeeze(smp(DLC(F,13),DLC(F,17)))  在scale轴maxpooling
            # pcm11 = self.cal_pcm(shift=11)
            pcm13 = self.cal_pcm(shift=13)
            pcm17 = self.cal_pcm(shift=17)
            out = torch.maximum(pcm13, pcm17)
            # out = torch.maximum(pcm11, torch.maximum(pcm13, pcm17))
            return out

        else:
            # DLC
            out = self.cal_pcm(shift=13)
            return out


    def cal_pcm(self, shift):
        """
        DLC
        :param cen: center feature map
        :param shift: shift size
        :return: MPCM map
        """
        B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(shift)
        s1 = (B1 - self.cen) * (B5 - self.cen)
        s2 = (B2 - self.cen) * (B6 - self.cen)
        s3 = (B3 - self.cen) * (B7 - self.cen)
        s4 = (B4 - self.cen) * (B8 - self.cen)   # transfer to tensor

        c12 = torch.minimum(s1, s2)
        c123 = torch.minimum(c12, s3)
        c1234 = torch.minimum(c123, s4)

        return c1234

    def circ_shift(self, shift):
        """
        cyclic shift compute for MPCM
        take each shift part and concatenate them according to the direction
        :param cen: center feature map  b*c*h*w
        :param shift: shift size
        :return:
        """
        _, _, h, w = self.cen.shape

        ###### B1 ######
        # (-d,-d) 左对角线
        B1_NW = self.cen[:, :, shift:, shift:]  # B1_NW is self.cen's SE
        B1_NE = self.cen[:, :, shift:, :shift]      # B1_NE is self.cen's SW
        B1_SW = self.cen[:, :, :shift, shift:]      # B1_SW is self.cen's NE
        B1_SE = self.cen[:, :, :shift, :shift]          # B1_SE is self.cen's NW
        B1_N = torch.cat([B1_NW, B1_NE], dim=3)
        B1_S = torch.cat([B1_SW, B1_SE], dim=3)
        B1 = torch.cat([B1_N, B1_S], dim=2)

        ###### B2 ######
        # (-d,0) 垂直
        B2_N = self.cen[:, :, shift:, :]  # B2_N is self.cen's S
        B2_S = self.cen[:, :, :shift, :]      # B2_S is self.cen's N
        B2 = torch.cat([B2_N, B2_S], dim=2)

        ###### B3 ######
        # (-d,d) 右对角线
        B3_NW = self.cen[:, :, shift:, w - shift:]  # B3_NW is self.cen's SE
        B3_NE = self.cen[:, :, shift:, :w-shift]      # B3_NE is self.cen's SW
        B3_SW = self.cen[:, :, :shift, w-shift:]      # B3_SW is self.cen's NE
        B3_SE = self.cen[:, :, :shift, :w-shift]          # B1_SE is self.cen's NW
        B3_N = torch.cat([B3_NW, B3_NE], dim=3)
        B3_S = torch.cat([B3_SW, B3_SE], dim=3)
        B3 = torch.cat([B3_N, B3_S], dim=2)

        ###### B4 ######
        # (0,d) 水平
        B4_W = self.cen[:, :, :, w - shift:]  # B2_W is self.cen's E
        B4_E = self.cen[:, :, :, :w-shift]          # B2_E is self.cen's S
        B4 = torch.cat([B4_W, B4_E], dim=3)

        ##### B5 ######
        B5_NW = self.cen[:, :, h - shift:, w - shift:]  # B5_NW is self.cen's SE
        B5_NE = self.cen[:, :, h-shift:, :w-shift]      # B5_NE is self.cen's SW
        B5_SW = self.cen[:, :, :h-shift, w-shift:]      # B5_SW is self.cen's NE
        B5_SE = self.cen[:, :, :h-shift, :w-shift]          # B5_SE is self.cen's NW
        B5_N = torch.cat([B5_NW, B5_NE], dim=3)
        B5_S = torch.cat([B5_SW, B5_SE], dim=3)
        B5 = torch.cat([B5_N, B5_S], dim=2)

        ##### B6 ######
        B6_N = self.cen[:, :, h - shift:, :]  # B6_N is self.cen's S
        B6_S = self.cen[:, :, :h-shift, :]      # B6_S is self.cen's N
        B6 = torch.cat([B6_N, B6_S], dim=2)

        ##### B7 ######
        B7_NW = self.cen[:, :, h - shift:, shift:]  # B7_NW is self.cen's SE
        B7_NE = self.cen[:, :, h-shift:, :shift]      # B7_NE is self.cen's SW
        B7_SW = self.cen[:, :, :h-shift, shift:]      # B7_SW is self.cen's NE
        B7_SE = self.cen[:, :, :h-shift, :shift]          # B7_SE is self.cen's NW
        B7_N = torch.cat([B7_NW, B7_NE], dim=3)
        B7_S = torch.cat([B7_SW, B7_SE], dim=3)
        B7 = torch.cat([B7_N, B7_S], dim=2)

        ##### B8 ######
        B8_W = self.cen[:, :, :, shift:]          # B8_W is self.cen's E
        B8_E = self.cen[:, :, :, :shift]          # B8_E is self.cen's S
        B8 = torch.cat([B8_W, B8_E], dim=3)

        return B1, B2, B3, B4, B5, B6, B7, B8





# class MPCMResNetFPNold(nn.Module):
#     def __init__(self, layers, channels, shift=3, classes=1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
#         """
#         samelayer = dlc, crosslayer = add
#         :param layers:
#         :param channels:
#         :param shift:
#         :param classes:
#         :param norm_layer:
#         :param norm_kwargs:
#         """
#         super(MPCMResNetFPNold, self).__init__()
#         self.layer_num = len(layers)
#
#         shift = shift
#
#         stem_width = int(channels[0])
#         self.stem = nn.Sequential()     # prefix="stem"
#         self.stem.add_module("bn0", norm_layer(stem_width, affine=False))
#         self.stem.add_module("conv1", nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False))
#         self.stem.add_module("bn1", norm_layer(stem_width))
#         self.stem.add_module("relu1", nn.ReLU())
#
#         self.stem.add_module("conv2", nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False))
#         self.stem.add_module("bn2", norm_layer(stem_width))
#         self.stem.add_module("relu2", nn.ReLU())
#
#         self.stem.add_module("conv3", nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False))
#         self.stem.add_module("bn3", norm_layer(stem_width*2))
#         self.stem.add_module("relu3", nn.ReLU())
#
#         self.head = FCNHead(in_channels=channels[1], channels=classes)  # look
#
#         self.layer1 = self._make_layer(block=BasicBlock, layers=layers[0], channels=channels[1], stride=1,
#                                        stage_index=1, in_channels=channels[1])
#
#         self.layer2 = self._make_layer(block=BasicBlock, layers=layers[1], channels=channels[2], stride=2,
#                                        stage_index=2, in_channels=channels[1])
#
#         self.layer3 = self._make_layer(block=BasicBlock, layers=layers[2], channels=channels[3], stride=2,
#                                        stage_index=3, in_channels=channels[2])
#
#         self.inc_c2 = nn.Sequential()
#         self.inc_c2.add_module("inc_c2_conv", nn.Conv2d(channels[2], channels[3], kernel_size=1, stride=1,
#                                                         padding=0, bias=False))
#         self.inc_c2.add_module("inc_c2_bn", norm_layer(channels[-1]))
#         self.inc_c2.add_module("inc_c2_relu", nn.ReLU())
#
#         self.inc_c1 = nn.Sequential()
#         self.inc_c1.add_module("inc_c1_conv", nn.Conv2d(channels[1], channels[3], kernel_size=1, stride=1,
#                                                         padding=0, bias=False))
#         self.inc_c1.add_module("inc_c1_bn", norm_layer(channels[-1]))
#         self.inc_c1.add_module("inc_c1_relu", nn.ReLU())
#
#     def _make_layer(self, block, layers, channels, stride, stage_index, in_channels, norm_layer=nn.BatchNorm2d,
#                     norm_kwargs=None):
#         layer = nn.Sequential()     # prefix="layer{}".format(stage_index)
#         downsample = (channels != in_channels) or (stride != 1)  # channels有变化或者stride不为1时，需要downsample
#
#         layer.add_module("stage{}".format(stage_index), block(inplanes=in_channels, planes=channels, stride=stride,
#                                                               downsample=downsample, norm_layer=norm_layer))
#
#         for _ in range(layers-1):  # 0，1，2
#             layer.add_module("stage{}".format(stage_index), block(inplanes=channels, planes=channels, stride=1,
#                                                                   downsample=None, norm_layer=norm_layer))
#
#         return layer
#
#     def forward(self, x):
#
#         _, _, orig_h, orig_w = x.shape
#         x = self.stem(x)    # 480*480*16
#
#         c1 = self.layer1(x)
#         _, _, c1_h, c1_w = c1.shape    # 480*480*16
#
#         c2 = self.layer2(c1)
#         _, _, c2_h, c2_w = c2.shape    # 240*240*32
#
#         c3 = self.layer3(c2)
#         _, _, c3_h, c3_w = c3.shape   # 120*120*64
#
#         c3pcm = self.cal_pcm(c3, shift=self.shift)
#         up_c3pcm = F.interpolate(c3pcm, size=(c2_h, c2_w), mode='bilinear', align_corners=False)
#
#         inc_c2 = self.inc_c2(c2)   # same channal size as c3 64
#         c2pcm = self.cal_pcm(c2, shift=self.shift)
#
#         c23pcm = up_c3pcm + c2pcm    # BLAM ？
#
#         up_c23pcm = F.interpolate(c23pcm, size=(c1_h, c1_w), mode='bilinear', align_corners=False)
#         inc_c1 = self.inc_c1(c1)
#         c1pcm = self.cal_pcm(c1, shift=self.shift)
#
#         out = up_c23pcm + c1pcm     # BLAM ？
#
#         # out size : 480*480*64
#         pred = self.head(out)
#         out = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
#
#         return out
#
#
#     def circ_shift(self, self.cen, shift):
#         """
#         cyclic shift compute for MPCM
#         take each shift part and concatenate them according to the direction
#         :param cen: center feature map  b*c*h*w
#         :param shift: shift size
#         :return:
#         """
#
#         _, _, h, w = cen.shape
#
#         ###### B1 ######
#         # (-d,-d) 左对角线
#         B1_NW = cen[:, :, shift:, shift:]  # B1_NW is cen's SE
#         B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
#         B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
#         B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
#         B1_N = torch.cat([B1_NW, B1_NE], dim=3)
#         B1_S = torch.cat([B1_SW, B1_SE], dim=3)
#         B1 = torch.cat([B1_N, B1_S], dim=2)
#
#         ###### B2 ######
#         # (-d,0) 垂直
#         B2_N = cen[:, :, shift:, :]  # B2_N is cen's S
#         B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
#         B2 = torch.cat([B2_N, B2_S], dim=2)
#
#         ###### B3 ######
#         # (-d,d) 右对角线
#         B3_NW = cen[:, :, shift:, w - shift:]  # B3_NW is cen's SE
#         B3_NE = cen[:, :, shift:, :w-shift]      # B3_NE is cen's SW
#         B3_SW = cen[:, :, :shift, w-shift:]      # B3_SW is cen's NE
#         B3_SE = cen[:, :, :shift, :w-shift]          # B1_SE is cen's NW
#         B3_N = torch.cat([B3_NW, B3_NE], dim=3)
#         B3_S = torch.cat([B3_SW, B3_SE], dim=3)
#         B3 = torch.cat([B3_N, B3_S], dim=2)
#
#         ###### B4 ######
#         # (0,d) 水平
#         B4_W = cen[:, :, :, w - shift:]  # B2_W is cen's E
#         B4_E = cen[:, :, :, :w-shift]          # B2_E is cen's S
#         B4 = torch.cat([B4_W, B4_E], dim=3)
#
#         ##### B5 ######
#         B5_NW = cen[:, :, h - shift:, w - shift:]  # B5_NW is cen's SE
#         B5_NE = cen[:, :, h-shift:, :w-shift]      # B5_NE is cen's SW
#         B5_SW = cen[:, :, :h-shift, w-shift:]      # B5_SW is cen's NE
#         B5_SE = cen[:, :, :h-shift, :w-shift]          # B5_SE is cen's NW
#         B5_N = torch.cat([B5_NW, B5_NE], dim=3)
#         B5_S = torch.cat([B5_SW, B5_SE], dim=3)
#         B5 = torch.cat([B5_N, B5_S], dim=2)
#
#         ##### B6 ######
#         B6_N = cen[:, :, h - shift:, :]  # B6_N is cen's S
#         B6_S = cen[:, :, :h-shift, :]      # B6_S is cen's N
#         B6 = torch.cat([B6_N, B6_S], dim=2)
#
#         ##### B7 ######
#         B7_NW = cen[:, :, h - shift:, shift:]  # B7_NW is cen's SE
#         B7_NE = cen[:, :, h-shift:, :shift]      # B7_NE is cen's SW
#         B7_SW = cen[:, :, :h-shift, shift:]      # B7_SW is cen's NE
#         B7_SE = cen[:, :, :h-shift, :shift]          # B7_SE is cen's NW
#         B7_N = torch.cat([B7_NW, B7_NE], dim=3)
#         B7_S = torch.cat([B7_SW, B7_SE], dim=3)
#         B7 = torch.cat([B7_N, B7_S], dim=2)
#
#         ##### B8 ######
#         B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
#         B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
#         B8 = torch.cat([B8_W, B8_E], dim=3)
#
#         return B1, B2, B3, B4, B5, B6, B7, B8
#
#     def cal_pcm(self, cen, shift):
#         """
#         MPCM Block
#         :param cen: center feature map
#         :param shift: shift size
#         :return: MPCM map
#         """
#         B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(cen, shift=shift)
#         s1 = (B1 - cen) * (B5 - cen)
#         s2 = (B2 - cen) * (B6 - cen)
#         s3 = (B3 - cen) * (B7 - cen)
#         s4 = (B4 - cen) * (B8 - cen)   # transfer to tensor
#
#         c12 = torch.minimum(s1, s2)
#         c123 = torch.minimum(c12, s3)
#         c1234 = torch.minimum(c123, s4)
#
#         return c1234


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