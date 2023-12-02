import numpy as np
from torch.utils.data import Dataset
import os
import torch
from PIL import Image, ImageOps, ImageFilter
import xml.etree.ElementTree as ET
import random


class SIRST(Dataset):
    NUM_CLASSES = 1
    def __init__(self, root, base_dir, split, mode, base_size, crop_size, transform=None, include_name=None):
        super(SIRST, self).__init__()

        # platform
        #####

        self.base_dir = base_dir     # open-sirst-v2
        self._root = os.path.join(root, self.base_dir)   # os.path.expanduser(root) 用于Linux上路径展开
        self.include_name = include_name  # 是否包含图片id

        self._split = split
        self.mode = mode

        self._transform = transform  # basic transform--totensor, normalize
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
            ids += [(root, line.strip()) for line in f.readlines()]

        random.shuffle(ids)

        return ids

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label_path = self._anno_path.format(*img_id)

        img = Image.open(img_path).convert('RGB')   # convert("L") ? 单通道
        if self.mode == 'test':
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            img = self._transform(img)

            return img   #, img_id[-1]


        mask = Image.open(label_path)
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
            # insert resize ?

        elif self.mode == 'val':    # resize + basic transform
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = img.resize((self.base_size, self.base_size), Image.NEAREST)

        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

        img, mask = self._transform(img), self._transform(mask)
        mask = mask.squeeze(0).astype('float32') / 255.0

        # if self.include_name:
        #     return img, mask, img_id[-1]

    @property
    def classes(self):
        return ('target')

    def _sync_transform(self, img, mask):
        """
        train/val dataset img and mask synchronized transform
        :param img:
        :param mask:
        :return:
        """
        random.seed(41)   # ****种子加在哪里的问题，调试
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        crop_size = self.crop_size
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

        return img, mask

    # dataloader 相关
    # def collate_fn(self):
