import numpy as np
from torch.utils.data import Dataset
import os
import torch
from PIL import Image, ImageOps, ImageFilter
import xml.etree.ElementTree as ET
import random


class SIRST(Dataset):
    NUM_CLASSES = 2
    def __init__(self, root, base_dir, split, mode, base_size, crop_size, transform=None, include_name=None):
        super(SIRST, self).__init__()

        # platform
        #####

        self.base_dir = base_dir     # open-sirst-v2
        self._root = os.path.join(root, self.base_dir)   # os.path.expanduser(root) 用于Linux上路径展开
        self.include_name = include_name  # 是否包含图片id

        self._split = split
        self.mode = mode

        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size

        if base_dir == 'open-sirst-v2':
            self._items = self._load_items(split)
            self._anno_path = os.path.join(self._root, 'annotations/masks/', '{}_pixels0.png')   # os.path.join(')
            self._image_path = os.path.join(self._root, 'images/targets/', '{}.png')
        else:
            raise ValueError('Unknown base_dir: {}'.format(base_dir))


    def _load_items(self, split):
        """
        load train/val/test sets items from txt file
        :param self:
        :param split: split to which datasets
        :return: list of images in select datasets
        """
        ids = []
        root = os.path.join(self._root, 'splits')
        lf = os.path.join(root, split + '.txt')
        with open(lf, 'r') as f:
            ids += [[line.strip()] for line in f.readlines()]

        random.shuffle(ids)
        return ids

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label_path = self._anno_path.format(*img_id)

        img = Image.open(img_path).convert('L')   # convert("RGB") 灰度图转RGB，RGB三通道复制

        if self.mode == 'test':
            img = self._img_transform(img)  # pil to array
            if self.transform is not None:
                img = self.transform(img)   # resize, normalize, toTensor
            # if self.include_name: + img_id[-1]
            return img


        mask = Image.open(label_path)
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)

        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)

        # if self.include_name: + img_id[-1]
        return img, mask


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
        """
        train/val dataset img and mask synchronized transform
        :param img:
        :param mask:
        :return:
        """
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
        target = np.array(mask).astype('int32')
        target[target > 0] = 1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        return ('background', 'target')

    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASSES

