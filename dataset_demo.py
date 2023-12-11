"""Base segmentation dataset"""
import random
import numpy as np
import os
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F



class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        """
        PIL to Array
        :param img:
        :return:
        """
        return np.array(img)

    def _mask_transform(self, mask):
        """
        PIL to Array
        :param mask:
        :return:
        """
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


class SIRST(SegmentationDataset):
    base_dir = "open-sirst-v2"
    def __init__(self, root='./data/', split='train', mode=None, transform=None, **kwargs):
        super(SIRST, self).__init__(root, split, mode, transform, **kwargs)
        self.root = os.path.join(root, self.base_dir)
        self.split = split
        self.mode = mode

        self._items = self._load_items(split)
        self._anno_path = os.path.join(self.root, 'annotations/masks/', '{}_pixels0.png')  # os.path.join(')
        self._image_path = os.path.join(self.root, 'images/targets/', '{}.png')

    def _load_items(self, split):
        """
        load train/val/test sets items from txt file
        :param self:
        :param split: split to which datasets
        :return: list of images in select datasets
        """
        ids = []
        root = os.path.join(self.root, 'splits')
        lf = os.path.join(root, split + '.txt')
        with open(lf, 'r') as f:
            ids += [[line.strip()] for line in f.readlines()]

        random.shuffle(ids)

        return ids


    def __getitem__(self, idx):
        img_id = self._items[idx]
        img = self._image_path.format(*img_id)
        mask = self._anno_path.format(*img_id)

        img = Image.open(img).convert('L')
        if self.mode == 'test':
            img = self._img_transform(img)   # pil to array
            if self.transform is not None:
                img = self.transform(img)    # resize, normalize, toTensor
            return img, img_id

        mask = Image.open(mask)
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask, img_id




    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('background', 'target')




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

        inter = (pred * target).sum(dim=(1, 2, 3))

        pred = pred.sum(dim=(1, 2, 3))
        target = target.sum(dim=(1, 2, 3))

        # sets_sum = pred + target
        # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        loss = (inter + self.smooth) / (pred + target - inter + self.smooth)
        loss = (1 - loss).mean()

        return loss








def validation(self):
    # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    is_best = False
    self.metric.reset()
    if self.args.distributed:
        model = self.model.module
    else:
        model = self.model
    torch.cuda.empty_cache()  # TODO check if it helps
    model.eval()
    for i, (image, target, filename) in enumerate(self.val_loader):
        image = image.to(self.device)
        target = target.to(self.device)

        with torch.no_grad():
            outputs = model(image)
        self.metric.update(outputs[0], target)
        pixAcc, mIoU = self.metric.get()
        logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

    new_pred = (pixAcc + mIoU) / 2
    if new_pred > self.best_pred:
        is_best = True
        self.best_pred = new_pred
    save_checkpoint(self.model, self.args, is_best)
    synchronize()




