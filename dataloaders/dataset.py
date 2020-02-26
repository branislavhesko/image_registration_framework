import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import Configuration
from utils.transforms import flatten


class RetinaDatasetPop2(Dataset):
    EXTENSION_FIRST = "png"
    EXTENSION_SECOND = "tif"

    def __init__(self, cfg: Configuration, transform=lambda x, y: (x, y)):
        self._images_first = []
        self._images_second = []
        self._cfg = cfg
        self._transform = transform
        self.load()

    def __len__(self):
        return len(self._images_first)

    def __getitem__(self, item):
        first = np.array(Image.open(self._images_first[item]))
        second = np.array(Image.open(self._images_second[item]))

        # TODO: remove
        first = cv2.resize(first, None, fx=0.5, fy=0.5)
        second = cv2.resize(second, None, fx=0.5, fy=0.5)
        first, second = self._transform(first, second)
        second = np.concatenate([second[:, :, np.newaxis]] * 3, axis=2)
        image_width = max(first.shape[1], second.shape[1])
        positive_pairs = self.find_positive_pairs(first, second, self._cfg.NUM_POS_PAIRS)
        positive_pairs = flatten(torch.from_numpy(np.array(positive_pairs)), image_width)
        return (
            torch.from_numpy(np.array(first) / 255.).permute([2, 0, 1]).float(),
            torch.from_numpy(np.array(second) / 255.).permute([2, 0, 1]).float(),
            torch.cat([positive_pairs.unsqueeze(dim=1), positive_pairs.unsqueeze(dim=1)], dim=1).long()
        )

    def load(self):
        images_first = sorted(glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_FIRST, "*." + self.EXTENSION_FIRST)))
        images_second = sorted(glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_SECOND, "*." + self.EXTENSION_SECOND)))

        for first in images_first:
            basename = os.path.basename(first)[:-4]
            second = list(filter(lambda x: basename in x, images_second))

            if len(second):
                self._images_first.append(first)
                self._images_second.append(second[0])

    @staticmethod
    def find_positive_pairs(first, second, num_pairs_needed):
        points = np.array((np.random.randint(0, first.shape[0] - 1, num_pairs_needed),
                           np.random.randint(0, first.shape[1] - 1, num_pairs_needed)))
        return points.astype(np.int32)


class VesselDataset(Dataset):

    def __init__(self, cfg: Configuration):
        self._cfg = cfg
        self.images = []
        self.masks = []
        self.load()

    def load(self):
        self.images = sorted(glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_FIRST, "*." + self._cfg.EXTENSION_FIRST)))
        self.masks = sorted(glob.glob(os.path.join(self._cfg.PATH_TO_MASKS, "*." + self._cfg.EXTENSION_MASK)))
        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.cvtColor(np.array(Image.open(self.images[item])), cv2.COLOR_GRAY2RGB) / 255.
        mask = cv2.imread(self.masks[item], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        print(f"{self.images[item]} and size: {np.mean(image)}")
        print(f"{self.masks[item]} and size: {np.mean(mask)}")
        return torch.from_numpy(image).permute([2, 0, 1]).float(), torch.from_numpy(mask).float()


if __name__ == "__main__":
    dataset = RetinaDatasetPop2(Configuration())
    dataset.load()
    print(dataset.find_positive_pairs(np.zeros((1000, 1000)), np.zeros((1000, 1000)), 20))