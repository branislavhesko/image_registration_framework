import cv2
import numpy as np
import torch
from scipy.ndimage import rotate


class Compose:

    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, image, mask):
        for transform in self._transforms:
            image, mask = transform(image, mask)
        return image, mask


class Rotate:

    def __init__(self, probability):
        self._prob = probability

    def __call__(self, image, mask):
        if np.random.rand() > self._prob:
            return image, mask
        angle = np.random.randint(-10, 10) + np.random.rand() - 0.5
        return rotate(image, angle, reshape=False), rotate(mask, angle, reshape=False, order=0)


class RandomContrast:
    def __init__(self, probability):
        self._prob = probability

    def __call__(self, image, mask):
        if np.random.rand() > self._prob:
            return image, mask
        contrast = 0.2 * (np.random.rand() - 0.5) + 1.
        return image * contrast, mask


class RandomCenterCrop:
    def __init__(self, probability, crop_range):
        assert crop_range < 1., "Crop range must be smaller than 1!"
        self._prob = probability
        self._crop_range = crop_range

    def __call__(self, image, mask):
        if np.random.rand() > self._prob:
            return image, mask
        shape = image.shape
        crop_size = np.random.randint(self._crop_range * shape[0], shape[0]), \
            np.random.randint(self._crop_range * shape[1], shape[1])
        start_point = np.random.randint(0, shape[0] - crop_size[0]), \
            np.random.randint(0, shape[1] - crop_size[1])
        return image[start_point[0]: crop_size[0], start_point[1]: crop_size[1], :], \
            mask[start_point[0]: crop_size[0], start_point[1]: crop_size[1]]


class Resize:

    def __init__(self, size):
        self._size = size

    def __call__(self, image, mask):
        return cv2.resize(image, self._size, interpolation=cv2.INTER_LINEAR), \
               cv2.resize(mask, self._size, interpolation=cv2.INTER_NEAREST)


class ToTensor:
    def __call__(self, image, mask):
        return torch.from_numpy(image).permute([2, 0, 1]).float(), torch.from_numpy(mask).int()


if __name__ == "__main__":
    image = np.random.rand(512, 512, 3)
    crop = Compose([
        RandomContrast(0.7),
        RandomCenterCrop(0.7, 0.8),
        Resize((566, 512)),
        Rotate(0.7),
        ToTensor()
    ])
    for i in range(100):
        print(crop(image, image[:, :, 0])[1].shape)