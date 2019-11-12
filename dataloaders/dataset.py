import glob
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

from config import Configuration


class RetinaDatasetPop2(Dataset):
    EXTENSION_FIRST = "png"
    EXTENSION_SECOND = "tif"

    def __init__(self, cfg: Configuration, transform=lambda x, y: (x, y)):
        self._images_first = []
        self._images_second = []
        self._cfg = cfg
        self._transform = transform

    def __len__(self):
        return len(self._images_first)

    def __getitem__(self, item):
        first = Image.open(self._images_first[item])
        second = Image.open(self._images_second[item])
        first, second = self._transform(first, second)
        positive_pairs = self.find_positive_pairs(first, second, self._cfg.NUM_POS_PAIRS)
        positive_pairs = torch.from_numpy(np.array(positive_pairs)).unsqueeze(dim=0)
        return (
            torch.from_numpy(np.array(first)).permute([2, 0, 1]),
            torch.from_numpy(np.array(second)).permute([2, 0, 1]),
            positive_pairs, positive_pairs
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
        center = np.array(first.shape[:2]) // 2
        points = (center // 2) * np.random.randn(num_pairs_needed, 2) + center
        return points.astype(np.int32)


if __name__ == "__main__":
    dataset = RetinaDatasetPop2(Configuration())
    dataset.load()
    print(dataset.find_positive_pairs(np.zeros((1000, 1000)), np.zeros((1000, 1000)), 20))