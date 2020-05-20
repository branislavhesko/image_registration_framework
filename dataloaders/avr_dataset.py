import glob

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from config import AVRConfiguration


class AVRDataset(Dataset):

    def __init__(self, config: AVRConfiguration, path_images, path_masks, transforms=lambda x, y: (x, y)):
        self._images = None
        self._masks = None
        self._config = config
        self._load(path_images, path_masks)
        self._transforms = transforms

    def _load(self, path_images, path_masks):
        self._images = glob.glob(path_images)
        self._masks = glob.glob(path_masks)

    def __len__(self):
        assert len(self._images) == len(self._masks)
        return len(self._images)

    def __getitem__(self, item):
        image = cv2.cvtColor(cv2.imread(self._images[item], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
        mask = cv2.imread(self._masks[item], cv2.IMREAD_COLOR)
        mask = self._process_mask(mask)
        image, mask = self._transforms(image, mask)
        return torch.from_numpy(image).permute([2, 0, 1]).float(), torch.from_numpy(mask).int()

    def _process_mask(self, mask):
        new_mask = np.zeros(mask.shape[:2])
        new_mask[(mask[:, :, 0] > 128) & (mask[:, :, 1] < 128) & (mask[:, :, 2] < 128)] = 1
        new_mask[(mask[:, :, 0] < 128) & (mask[:, :, 1] < 128) & (mask[:, :, 2] > 128)] = 2
        return new_mask
