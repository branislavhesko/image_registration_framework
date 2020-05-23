import cv2
import numpy as np
import torch

from dataloaders.dataset import VesselDataset


class AVRDataset(VesselDataset):

    def __len__(self):
        assert len(self.images) == len(self.masks)
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.cvtColor(cv2.imread(self.images[item], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
        mask = cv2.imread(self.masks[item], cv2.IMREAD_COLOR)
        mask[mask > 5] = 255
        mask = cv2.dilate(mask, np.ones((5, 5)))
        image, mask = [cv2.resize(i, self._cfg.SIZE, interpolation=cv2.INTER_NEAREST) for i in [image, mask]]
        mask = self._process_mask(mask)
        mask[np.mean(image, axis=2) < 0.01] = 3
        return torch.from_numpy(image).permute([2, 0, 1]).float(), torch.from_numpy(mask).int()

    def _process_mask(self, mask):
        new_mask = np.zeros(mask.shape[:2], dtype=np.int32)
        new_mask[(mask[:, :, 0] > 128) & (mask[:, :, 1] < 128) & (mask[:, :, 2] < 128)] = 1
        new_mask[(mask[:, :, 0] < 128) & (mask[:, :, 1] < 128) & (mask[:, :, 2] > 128)] = 2
        return new_mask
