import numpy as np

import torch.nn as nn
import torch

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc




def parse_args():
    """
    Training option and segmentation experiments
    :return:
    """
    parser = argparse.ArgumentParser(description='alcnet pytorch')

    ####### model #######
    parser.add_argument('--net-choice', type=str, default='ALCNet', help='model')
    parser.add_argument('--pyramid-mode', type=str, default='Dec', help='Inc,Dec') # ?
    parser.add_argument('--r', type=int, default=2, help='choice:1,2,4')   # ?
    parser.add_argument('--summary', action='store_true', default=False, help='print parameters')   # 命令行输入参数则为True(激活)，否则为False
    parser.add_argument('--scale-mode', type=str, default='Multiple', help='choice:Single, Multiple, Selective')
    parser.add_argument('--pyramid-fuse', type=str, default='bottomuplocal', help='choice:add, max, sk')
    parser.add_argument('--cue', type=str, default='lcm', help='choice:lcm, orig')  # ?

    ####### dataset #######
    parser.add_argument('--data_root', type=str, default='./data/', help='dataset path')
    parser.add_argument('--out', type=str, default='./', help='metrics saved path')
    parser.add_argument('--dataset', type=str, default='open-sirst-v2', help='choice:DENTIST, Iceberg')
    parser.add_argument('--workers', type=int, default=1, metavar='N', help='dataloader threads')   # metavar ?
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--blocks', type=int, default=4, help='[1] * blocks')
    parser.add_argument('--channels', type=int, default=16, help='channels')
    parser.add_argument('--shift', type=int, default=13, help='shift')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='iou threshold')
    parser.add_argument('--train-split', type=str, default='train_v1', help='choice:train, trainval')
    parser.add_argument('--val-split', type=str, default='val_v1', help='choice:test, val')


    ####### training hyperparameters #######
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epoch')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,200', help='epochs at which learning rate decays (default: 40,60)')
    parser.add_argument('--gamma', type=int, default=2, help='gamma for Focal Soft Iou Loss')
    parser.add_argument('--lambda', type=int, default=1, help='lambda for TV Soft Iou Loss')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='weight-decay')
    parser.add_argument('--no-wd', action='store_true', default=False, help='whether to remove weight decay on bias and beta/gamma for bn layers')
    parser.add_argument('--sparsity', action='store_true', help='whether to use sparsity regularization')   # ？
    parser.add_argument('--score-thresh', type=float, default=0.5, help='score-thresh')

    ####### cuda and logging #######
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--gpus', type=str, default='0', help='Training with which gpus like 0,1,2,3')
    parser.add_argument('--kvstore', type=str, default='device', help='kvstore to use for trainer/module.')  # multi-GPU training
    parser.add_argument('--dtype', type=str, default='float32', help='data type for training')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay rate')  # ? 与上边weight-decay有什么区别
    parser.add_argument('--log-dir', type=str, default='./logs', help='log directory')
    parser.add_argument('--log-iter', type=int, default=10, help='print log every log-iter')



    ####### checking point #######
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')  # './params/
    parser.add_argument('--save-dir', type=str, default='./params', help='Directory for saving checkpoint models')
    parser.add_argument('--colab', action='store_true', help='whether using colab')

    ####### evaluation #######
    parser.add_argument('--eval', action='store_true', default=False, help='evaluating only')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--metric', type=str, default='mAP', help='choich:F1, IoU, mAP')

    ####### synchronized BatchNorm for multiple devices or distributed system #######
    parser.add_argument('--syncbn', action='store_true', help='using Synchronized Cross-GPU BatchNorm')

    args = parser.parse_args()


    ## used devices  (ctx)
    # available_gpus = list(range(torch.cuda.device_count()))
    if args.no_cuda or (torch.cuda.is_available() == False):
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = torch.device('cpu')
    else:
        args.ctx = [torch.device('cuda:' + i) for i in args.gpus.split(',') if i.strip()]
        print('Using {} GPU: {}, '.format(len(args.ctx), args.ctx))

    ## Synchronized BatchNorm setting
    args.norm_layer = nn.SyncBatchNorm if args.syncbn else nn.BatchNorm2d
    args.norm_kwargs = {'num_devices': len(args.ctx)} if args.syncbn else {}

    print(args)
    return args



### rewrite model

from torchvision.models.resnet import BasicBlock, Bottleneck
import torch.nn.functional as F

class backbone(nn.Module):
    """
    defalut Resnet
    layers = [args.blocks] * stage_num , the num of basickblock/bottleneck in each stage
    stage_num  default:3
    """
    def __init__(self, layers, channels):
        super(backbone, self).__init__()

        # 1->16
        self.conv1 = nn.Conv2d(1, channels[0] * 2 , kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0] * 2)
        self.relu = nn.ReLU(inplace=True)

        self.layer_num = len(layers)

        self.layer1 = self._make_layer(block=BasicBlock, layers=layers[0], channels=channels[1], stride=1,
                                       stage_index=1, in_channels=channels[1])

        self.layer2 = self._make_layer(block=BasicBlock, layers=layers[1], channels=channels[2], stride=2,
                                       stage_index=2, in_channels=channels[1])

        self.layer3 = self._make_layer(block=BasicBlock, layers=layers[2], channels=channels[3], stride=2,
                                       stage_index=3, in_channels=channels[2])



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


class BLAM(nn.Module):
    """
    Bottum Up Local Attention Module for feature map fusion
    channels: input channel num
    r: reduction ratio
    """
    def __init__(self, channels=64, r=2):
        super(BLAM, self).__init__()
        inter_channels = int(channels // r)

        self.bn0 = nn.BatchNorm2d(channels)

        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=1)
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








############# old model.py #############
########################################
########################################

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
        self.stem.add_module("bn0", norm_layer(1, affine=False))   # inchannels的选择 1/3
        self.stem.add_module("conv1", nn.Conv2d(1, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False))
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





