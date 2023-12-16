import torch
import torch.nn as nn

class PCMLayer(nn.Module):
    def __init__(self, mpcm=True):
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
        :param shift: shift size (11/13/17)
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