import numpy as np
import torch.nn as nn
import torch
# from skimage import measure
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos = np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)

        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh)

    # detach and tonumpy
    predict = predict.detach().numpy()
    target = target.detach().numpy()

    if len(target.shape) == 3:
        target = np.expand_dims(target.astype(float), axis=1)
    elif len(target.shape) == 4:
        target = target.astype(float)
    else:
        raise ValueError("Unknown target dimension")

    predict = predict.astype(float)
    intersection = predict * ((predict == target).astype(float))

    tp = intersection.sum()
    fp = (predict * ((predict != target).astype(float))).sum()
    tn = ((1 - predict) * ((predict == target).astype(float))).sum()
    fn = (((predict != target).astype(float)) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp+fp

    return tp, pos, fp, neg, class_pos



class mIoU:
    def __init__(self, nclass):
        self.nclass = nclass
        self.reset()

    def update(self, pred, labels):
        # detach and tonumpy
        pred = pred.detach().numpy()
        labels = labels.detach().numpy()

        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


    def batch_pix_accuracy(self, output, target):

        # shape check
        if len(target.shape) == 3:
            target = np.expand_dims(target.astype(float), axis=1)
        elif len(target.shape) == 4:
            target = target.astype(float)
        else:
            raise ValueError("Unknown target dimension")

        assert output.shape == target.shape, "Predict and Label Shape Don't Match"

        predict = (output > 0).astype(float)  # P  x>0,sigmoid>0.5->true
        pixel_labeled = (target > 0).astype(float).sum()   # T
        pixel_correct = (((predict == target).astype(float)) * ((target > 0).astype(float))).sum()  # TP

        assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):

        mini = 1
        maxi = 1
        nbins = 1

        if len(target.shape) == 3:
            target = np.expand_dims(target.astype(float), axis=1)
        elif len(target.shape) == 4:
            target = target.astype(float)
        else:
            raise ValueError("Unknown target dimension")

        predict = (output > 0).astype(float)  # P
        intersection = predict * ((predict == target).astype(float))

        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter

        assert (area_inter <= area_union).all(), \
            "Error: Intersection area should be smaller than Union area"
        return area_inter, area_union




class nIoU:
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, pred, labels):
        # detach and tonumpy
        pred = pred.detach().numpy()
        labels = labels.detach().numpy()

        inter_arr, union_arr = self.batch_intersection_union(pred, labels)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])


    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        if len(target.shape) == 3:
            target = np.expand_dims(target.astype(float), axis=1)
        elif len(target.shape) == 4:
            target = target.astype(float)
        else:
            raise ValueError("Unknown target dimension")

        predict = (output > 0).astype(float)  # P
        intersection = predict * ((predict == target).astype(float))

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr







# class PD_FA():
#     def __init__(self, nclass, bins):
#         super(PD_FA, self).__init__()
#         self.nclass = nclass
#         self.bins = bins
#         self.image_area_total = []
#         self.image_area_match = []
#         self.FA = np.zeros(self.bins+1)
#         self.PD = np.zeros(self.bins + 1)
#         self.target= np.zeros(self.bins + 1)
#     def update(self, preds, labels):
#
#         for iBin in range(self.bins+1):
#             score_thresh = iBin * (255/self.bins)
#             predits = np.array((preds > score_thresh).cpu()).astype('int64')
#             predits = np.reshape (predits, (256, 256))
#             labelss = np.array((labels).cpu()).astype('int64') # P
#             labelss = np.reshape(labelss, (256, 256))
#
#             image = measure.label(predits, connectivity=2)
#             coord_image = measure.regionprops(image)
#             label = measure.label(labelss , connectivity=2)
#             coord_label = measure.regionprops(label)
#
#             self.target[iBin] += len(coord_label)
#             self.image_area_total = []
#             self.image_area_match = []
#             self.distance_match = []
#             self.dismatch = []
#
#             for K in range(len(coord_image)):
#                 area_image = np.array(coord_image[K].area)
#                 self.image_area_total.append(area_image)
#
#             for i in range(len(coord_label)):
#                 centroid_label = np.array(list(coord_label[i].centroid))
#                 for m in range(len(coord_image)):
#                     centroid_image = np.array(list(coord_image[m].centroid))
#                     distance = np.linalg.norm(centroid_image - centroid_label)
#                     area_image = np.array(coord_image[m].area)
#                     if distance < 3:
#                         self.distance_match.append(distance)
#                         self.image_area_match.append(area_image)
#
#                         del coord_image[m]
#                         break
#
#             self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
#             self.FA[iBin] += np.sum(self.dismatch)
#             self.PD[iBin]+=len(self.distance_match)
#
#     def get(self, img_num):
#
#         Final_FA = self.FA / ((256 * 256) * img_num)
#         Final_PD = self.PD /self.target
#
#         return Final_FA, Final_PD
#
#
#     def reset(self):
#         self.FA = np.zeros([self.bins+1])
#         self.PD = np.zeros([self.bins+1])