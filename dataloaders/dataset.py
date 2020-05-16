import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import ArteryVeinConfiguration, Configuration
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
        first = cv2.resize(first, None, fx=0.25, fy=0.25)
        second = cv2.cvtColor(cv2.resize(second, None, fx=0.25, fy=0.25), cv2.COLOR_GRAY2RGB)
        first, second = self._transform(first, second)
        image_width = max(first.shape[1], second.shape[1])
        positive_pairs = self.find_positive_pairs(first, second, self._cfg.NUM_POS_PAIRS)
        positive_pairs = flatten(torch.from_numpy(np.array(positive_pairs)), image_width)
        return (
            torch.from_numpy(np.array(first) / 255.).permute([2, 0, 1]).float(),
            torch.from_numpy(np.array(second) / 255.).permute([2, 0, 1]).float(),
            torch.cat([positive_pairs.unsqueeze(dim=1), positive_pairs.unsqueeze(dim=1)], dim=1).long()
        )

    def load(self):
        images_second = sorted(glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_FIRST, "*." + self._cfg.EXTENSION_FIRST)))
        images_first = sorted(glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_SECOND, "*." + self._cfg.EXTENSION_SECOND)))

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


class ArteryVeinDataset(VesselDataset):
    COLORS = ({255, 0, 0}, {0, 0, 255}, {100, 100, 100})

    @staticmethod
    def read_transparent_png(filename):
        image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if image_4channel.shape[-1] == 3:
            return image_4channel[:, :, ::-1]
        alpha_channel = image_4channel[:, :, 3]
        rgb_channels = image_4channel[:, :, :3]

        # White Background Image
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

        # Alpha factor
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

        # Transparent Image Rendered on White Background
        base = rgb_channels.astype(np.float32) * alpha_factor
        white = white_background_image.astype(np.float32) * (1 - alpha_factor)
        final_image = base + white
        return final_image.astype(np.uint8)

    def load(self):
        sub_folders = next(os.walk(self._cfg.PATH_TO_IMAGES_FIRST))[1]
        for sub_folder in sub_folders:
            image = glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_FIRST, sub_folder, "*reg.png"))
            mask = glob.glob(os.path.join(self._cfg.PATH_TO_IMAGES_FIRST, sub_folder, "*reg_AV.png"))
            if any(image) and any(mask):
                self.images.append(image[0])
                self.masks.append(mask[0])
                print(f"Image: {image} with corresponding mask found.")

    def __getitem__(self, item):
        print("Processing image: {}, mask: {}".format(self.images[item], self.masks[item]))
        image = np.array(Image.open(self.images[item])) / 255.
        mask = self.read_transparent_png(self.masks[item])
        label = np.zeros(mask.shape[:2])
        label[(mask[:, :, 0] > 250) & (mask[:, :, 1] < 10) & (mask[:, :, 2] < 10)] = 1
        label[(mask[:, :, 0] < 10) & (mask[:, :, 1] < 10) & (mask[:, :, 2] > 250)] = 2
        label[(mask[:, :, 0] > 250) & (mask[:, :, 1] < 10) & (mask[:, :, 2] > 250)] = 3
        return torch.from_numpy(image).permute([2, 0, 1]).float(), torch.from_numpy(label).long()


if __name__ == "__main__":
    dataset = ArteryVeinDataset(ArteryVeinConfiguration())
    dataset.load()
    from matplotlib import pyplot as plt
    for index in range(len(dataset)):
        _, mask = dataset[index]
        print(index)
        plt.imshow(mask.numpy())
        plt.show()