from typing import Iterable

import numpy as np
import torch
from skimage.transform import rotate


def flatten(pair_indices, num_cols):
    """
    Pair indices are in form [N x 2], with [row, column form]
    :param pair_indices: -
    :param num_cols: number of columns in matrix
    :return:
    """
    return pair_indices[0, :] * num_cols + pair_indices[1, :]


def unflatten(pair_indices_flat, num_cols):
    return torch.cat([(pair_indices_flat // num_cols).unsqueeze(dim=1),
                      (pair_indices_flat % num_cols).unsqueeze(dim=1)], dim=1)


class Compose:

    def __init__(self, transforms: Iterable):
        self._transforms = transforms

    def __call__(self, image, points):
        for transform in self._transforms:
            image, points = transform(image, points)


class Rotate:

    def __init__(self, max_angle):
        self._angle = max_angle

    def __call_(self, image, points):
        random_angle = np.random.randint(-self._angle, self._angle, [1])
        rotated = rotate(image, random_angle)
        random_angle_rad = deg_to_rad(random_angle)
        rotation_matrix = np.array([[np.cos(random_angle_rad), -np.sin(random_angle_rad)],
                                    [np.sin(random_angle_rad), np.cos(random_angle_rad)]])
        return rotated, np.matmul(points, rotation_matrix)


class ContrastTransform:
    pass


def deg_to_rad(angle):
    return angle * np.pi / 360.
