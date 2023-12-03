import torchnet.meter.meter as m
import torchnet.logger as logger
import threading
import numpy as np

class PixAccMeter(m.Meter):
    """Computes pixAcc
    """

    def __init__(self, nclass):
        super(PixAccMeter, self).__init__()
        self.reset()

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def add(self, output=None, target=None, weight=None):
        """
        compute batch pixelwise accuracy
        :param output: pred, tensor, 4D(B, C, H, W), int64, 0 or 1
        :param target: label, tensor, 3D(B, H, W), int64, 0 or 1
        :return:
        """
        """PixAcc"""
        # inputs are NDarray, output 4D, target 3D
        # the category 0 is ignored class, typically for background / boundary
        # predict = np.argmax(output.asnumpy(), 1).astype('int64')
        # print("Metric output.shape: ", output.shape)
        # print("Metric target.shape: ", target.shape)
        # print("output.max(): ", output.max().asscalar())
        # print("target.max(): ", target.max().asscalar())
        if len(target.shape) == 3:
            target = target.squeeze(1).asnumpy().astype('int64')  # B,1,H,W    # astype('int64')? 改int8
        elif len(target.shape) == 4:
            target = target.asnumpy().astype('int64')  # T = TP + FN
        else:
            raise ValueError("Unknown target dimension")
        # print("output.shape: ", output.shape)
        # print("target.shape: ", target.shape)
        assert output.shape == target.shape, "Predict and Label Shape need to Match"
        predict = (output.asnumpy() > 0).astype('int64')  # P = TP + FP
        pixel_labeled = np.sum(target > 0)  # T = TP + FN
        pixel_correct = np.sum((predict == target) * (target > 0))  # TP

        assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
        return pixel_correct, pixel_labeled

        def value(self):
            """Returns pixel accuracy"""
            self.correct = self.total_correct
