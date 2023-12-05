"""Evaluation Metrics for Semantic Segmentation of Foreground Only"""
import torch
from torchnet.meter import meter, ConfusionMeter, TimeMeter
import torchnet.logger as logger
import threading
import numpy as np
from torch.nn.functional import sigmoid


def confusion_matrix(output, target, threshold=0.5, class_num=2):
    """
    compute confusion matrix
    :param output: model output, tensor, B,C,H,W
    :param target: label, tensor, B,H,W
    :param class_num: 2, target and background
    :return: list of confusion matrix of each image in a batch
    """
    confusion_m = []  # list
    cm = ConfusionMeter(class_num, normalized=False)
    img_per_batch = output.shape[0]
    output = torch.squeeze(output, 1)  # B,H,W
    for b in range(img_per_batch):
        predict = (sigmoid(output[b]) > threshold).type(torch.int8)
        assert predict.shape == target[b].shape, "Predict and Label Shape need to Match"
        cm.add(predict.view(-1), target[b].view(-1))
        confusion_m.append(cm.value())
    return confusion_m

class IoUMetric(meter.Meter):
    def __init__(self, select='IoU'):
        super(IoUMetric, self).__init__()
        self.select = select
        self.reset()

    def reset(self):
        self.IoU = 0
        self.nIoU = 0

    def add(self, confusion_m):
        """

        :param confusion_m: list of confusion matrix of each image in a batch
        :return:
        """
        confusion_m = np.array(confusion_m)
        if self.select == 'nIoU':
            iou_b = []
            for each in confusion_m:
                iou = each[1, 1] / (each[1].sum() + each[:, 1].sum() + each[1, 1])
                iou_b.append(iou)
            self.nIoU = np.mean(iou_b)
        elif self.select == 'IoU':
            tp = confusion_m[:, 1, 1].sum()
            t = confusion_m[:, 1, :].sum()
            p = confusion_m[:, :, 1].sum()
            self.IoU = 1.0 * tp / (t + p - tp + np.spacing(1))
        else:
            raise ValueError("Unknown IoU select: {}".format(self.select))
    def value(self):
        if self.select == 'nIoU':
            return self.nIoU
        elif self.select == 'IoU':
            return self.IoU
        else:
            raise ValueError("Unknown IoU select: {}".format(self.select))

class PFMetric(meter.Meter):
    def __init__(self, select):
        super(PFMetric, self).__init__()
        self.select = select
        self.reset()

    def reset(self):
        self.pd = 0
        self.fa = 0

    def add(self, confusion_m):
        if self.select == 'Pd' or self.select == 'ROC':
            self.pd = confusion_m[:, 1, 1].sum() / confusion_m[:, 1, :].sum()
        elif self.select == 'Fa' or self.select == 'ROC':
            self.fa = confusion_m[:, 0, 1].sum() / confusion_m[:, 0, :].sum()
        else:
            raise ValueError("Unknown PdFa select: {}".format(self.select))

    def value(self):
        if self.select == 'Pd':
            return self.pd
        elif self.select == 'Fa':
            return self.fa
        elif self.select == 'ROC':
            return self.pd, self.fa
        else:
            raise ValueError("Unknown PdFa select: {}".format(self.select))


