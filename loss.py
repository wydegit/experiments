import torch.nn as nn

class SoftIoULoss(nn.Module):
    def __init__(self, smooth=.1):
        """
        custom loss function -- SoftIoU Loss
        """
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        assert pred.size() == target.size()
        assert pred.dim() == 3

        pred = self.sigmoid(pred)

        inter = (pred * target).sum(dim=(1, 2))

        pred = pred.sum(dim=(1, 2))
        target = target.sum(dim=(1, 2))

        # sets_sum = pred + target
        # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        loss = (inter + self.smooth) / (pred + target - inter + self.smooth)
        loss = (1 - loss).mean()

        return loss

