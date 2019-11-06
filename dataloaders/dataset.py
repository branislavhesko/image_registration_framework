from torch.utils.data import Dataset
import torch

from config import Configuration


class RetinaDatasetPop2(Dataset):

    def __init__(self, cfg: Configuration):
        self._images = []

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        return None