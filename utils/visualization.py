import numpy as np
import torch
from sklearn.decomposition import PCA

from utils.transforms import unflatten


def visualize_points(image1: np.ndarray, image2: np.ndarray, points: np.ndarray):
    """
    Show point-pairs together with images in a common figure.
    :param image1: (M x N x C matrix)
    :param image2: (M x N x C matrix)
    :param points: (A x 2 matrix)
    :return: figure
    """

    pairs1 = unflatten(points[:, 0], image1.shape[1])
    pairs2 = unflatten(points[:, 1], image2.shape[1])
    image1[pairs1[:, 0], pairs1[:, 1], :] = np.array((255, 0, 0))
    image2[pairs2[:, 0], pairs2[:, 1], :] = np.array((0, 255, 0))
    return image1, image2


def visualize_features(features1: np.ndarray, features2: np.ndarray):
    pca = PCA(n_components=3)
    h, w, ch = features1.shape
    feats1_reduced = pca.fit_transform(features1.reshape(-1, ch))
    feats2_reduced = pca.fit_transform(features2.reshape(-1, ch))
    feats1_reduced_n = (feats1_reduced - np.amin(feats1_reduced)) / (np.amax(feats1_reduced) - np.amin(feats1_reduced))
    feats2_reduced_n = (feats2_reduced - np.amin(feats2_reduced)) / (np.amax(feats2_reduced) - np.amin(feats2_reduced))
    return torch.from_numpy(feats1_reduced_n.reshape(h, w, 3)).permute([2, 0, 1]), \
           torch.from_numpy(feats2_reduced_n.reshape(h, w, 3)).permute([2, 0, 1])
