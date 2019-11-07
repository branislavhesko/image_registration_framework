import glob
import numpy as np
import os
from torch.utils.data import Dataset
import torch

from config import Configuration


class RetinaDatasetPop2(Dataset):
    EXTENSION_FIRST = "png"
    EXTENSION_SECOND = "tif"

    def __init__(self, cfg: Configuration):
        self._images_first = []
        self._images_second = []
        self._cfg = cfg

    def __len__(self):
        return len(self._images_first)

    def __getitem__(self, item):
        return None

    def load(self):
        images_first = sorted(glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_FIRST, "*." + self.EXTENSION_FIRST)))
        images_second = sorted(glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_SECOND, "*." + self.EXTENSION_SECOND)))

        for first in images_first:
            basename = os.path.basename(first)[:-4]
            second = list(filter(lambda x: basename in x, images_second))

            if len(second):
                self._images_first.append(first)
                self._images_second.append(second[0])

    def find_positive_pairs(self, first, second, num_pairs_needed):
        center = np.array(first.shape[:2]) // 2
        points = (center // 2) * np.random.randn(num_pairs_needed, 2) + center
        return points.astype(np.int32)


if __name__ == "__main__":
    dataset = RetinaDatasetPop2(Configuration())
    dataset.load()
    print(dataset.find_positive_pairs(np.zeros((1000, 1000)), np.zeros((1000, 1000)), 20))