import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASKCResNetFPN(nn.Module):
    def __init__(self, layers, channels, fuse_mode, act_dilation, classes=1, tinyFlag=False,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(ASKCResNetFPN, self).__init__()

        self.layer_num = len(layers)
        self.tinyFlag = tinyFlag

        stem_width = int(channels[0])

        self.stem = nn.Sequential() # prefix="stem"
        # 数据输入前先预处理一下
        self.stem.add_module("bn0", norm_layer(stem_width, affine=False))

        #
        if tinyFlag:
            self.stem.add_module("conv1", nn.Conv2d(3, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False))
            self.stem.add_module("bn1", norm_layer(stem_width*2))
            self.stem.add_module("relu1", nn.ReLU())

        else:
            self.stem.add_module("conv1", nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False))
            self.stem.add_module("bn1", norm_layer(stem_width))
            self.stem.add_module("relu1", nn.ReLU())

            self.stem.add_module("conv2", nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False))
            self.stem.add_module("bn2", norm_layer(stem_width))
            self.stem.add_module("relu2", nn.ReLU())

            self.stem.add_module("conv3", nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False))
            self.stem.add_module("bn3", norm_layer(stem_width*2))
            self.stem.add_module("relu3", nn.ReLU())
            self.stem.add_module("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # FCN的head
        self.head =



    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, norm_layer=nn.BatchNorm2d,
                    norm_kwargs=None):
        layer = nn.Sequential()  # prefix="layer{}".format(stage_index)
        downsample = (channels != in_channels) or (stride != 1)  # channels有变化或者stride不为1时，需要downsample
        layer.add_module("block0", block(in_channels, channels, stride, downsample, norm_layer,
                                          norm_kwargs=norm_kwargs, prefix=''))
        for _ in range(layers-1):
            layer.add_module(block(channels, stride=1, in_channels=channels, norm_layer=norm_layer, prefix='')

        return layer

    def _fuse_layer(self, fuse_mode, channels, act_dilation):
        if fuse_mode == "Direct_Add"
            fuse_layer =