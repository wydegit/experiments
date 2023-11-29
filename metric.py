"""Evaluation Metrics for Semantic Segmentation of Foreground Only"""
import threading


import numpy as np
import torch
import torchnet.meter as meter


import torchnet.meter as meter

# torchnet.meter.Meter和meterlogger中 Visualize, reset update的重写

__all__ = ['SigmoidMetric', 'batch_pix_accuracy', 'batch_intersection_union']

# 用torchnet.meter库重写

class SigmoidMetric(meter):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SigmoidMetric, self).__init__('pixAcc & mIoU')
        self.nclass = nclass
        self.lock = threading.Lock()   # ?
        self.reset()    # ?

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """

        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(pred, label)  # correct:TP, labeled:TP+FN
            inter, union = batch_intersection_union(pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union

        if isinstance(preds, np.ndarray):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker, args=(self, label, pred),)
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        # print("self.total_correct: ", self.total_correct)
        # print("self.total_label: ", self.total_label)
        # print("self.total_union: ", self.total_union)
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
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
        target = target.squeeze(1).asnumpy().astype('int64')          # B,1,H,W    # astype('int64')? 改int8
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


def batch_intersection_union(output, target, nclass):
    """
    compute batch mIoU
    :param output: pred 4D(B, C, H, W), int64, 0 or 1
    :param target: label 3D(B, H, W), int64, 0 or 1
    :param nclass: class
    :return:
    """
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    predict = (output.asnumpy() > 0).astype('int64')  # P
    if len(target.shape) == 3:
        target.unsqueeze(1).asnumpy().astype('int64')  # T  (B,1,H,W)
    elif len(target.shape) == 4:
        target = target.asnumpy().astype('int64')  # T
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * (predict == target)  # TP

    # areas of intersection and union   (np.histogram的用法)
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))  # 统计intersection里为1的
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union