import numpy as np
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import random

class SIRST(Dataset)

    NUM_CLASS = 1

    def __init__(self, base_dir, root, split, mode=None, transform=None, include_name=None):
        super(SIRST, self).__init__()

