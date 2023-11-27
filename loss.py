import torch
import torch.nn as nn

class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1):
        """
        custom loss function -- SoftIoU Loss
        """
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        pred = self.sigmoid(pred)

        intersection = pred * target
        loss = (intersection.sum() + self.smooth) / (pred.sum() + target.sum() - intersection.sum() + self.smooth)

        loss = 1 - loss.mean()

        return loss

class SamplewiseSoftIoULoss(nn.Module):
    def __init__(self, smooth=.1):
        super(SamplewiseSoftIoULoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        pred = self.sigmoid(pred)

        intersection = (pred * target).sum(dim=(1, 2, 3))
        pred = pred.sum(dim=(1, 2, 3))
        target = target.sum(dim=(1, 2, 3))

        loss = (intersection + self.smooth) / (pred + target - intersection + self.smooth)
        loss = (1 - loss).mean()

        return loss