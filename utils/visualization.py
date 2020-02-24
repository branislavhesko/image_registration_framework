import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from utils.transforms import unflatten


def visualize_closest_points(image1, image2, features1, features2, number_of_correspondences=100):
    random_points = (
        np.random.randint(0, image1.shape[1], size=number_of_correspondences),
        np.random.randint(0, image1.shape[0], size=number_of_correspondences))
    correspondences = []
    for idx in range(number_of_correspondences):
        distance_point_second_image = np.sqrt(np.sum(
            np.power(features2 - features1[random_points[1][idx], random_points[0][idx], :], 2), axis=2))
        correspondences.append(np.unravel_index(np.argmin(
            distance_point_second_image), distance_point_second_image.shape)[::-1])
    fig, ax = plt.subplots(1, 2)
    correspondences = np.asarray(correspondences).T
    ax[0].imshow(image1)
    ax[0].plot(*np.asarray(random_points)[:], "+r")
    for i in range(number_of_correspondences):
        ax[0].text(random_points[0][i] + 5, random_points[1][i] - 5, str(i))
        ax[1].text(correspondences[0][i] + 5, correspondences[1][i] - 5, str(i))
    ax[1].imshow(image2)
    ax[1].plot(*correspondences, "+b")
    return fig


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

#
# def visualize_features(features1: np.ndarray, features2: np.ndarray):
#     pca = PCA(n_components=3)
#     h, w, ch = features1.shape
#     feats1_reduced = pca.fit_transform(features1.reshape(-1, ch))
#     feats2_reduced = pca.fit_transform(features2.reshape(-1, ch))
#     feats1_reduced_n = (feats1_reduced - np.amin(feats1_reduced)) / (np.amax(feats1_reduced) - np.amin(feats1_reduced))
#     feats2_reduced_n = (feats2_reduced - np.amin(feats2_reduced)) / (np.amax(feats2_reduced) - np.amin(feats2_reduced))
#     return torch.from_numpy(feats1_reduced_n.reshape(h, w, 3)).permute([2, 0, 1]), \
#            torch.from_numpy(feats2_reduced_n.reshape(h, w, 3)).permute([2, 0, 1])


def visualize_features(features_iterable):
    pca = PCA(n_components=3)
    output = []
    for features in features_iterable:
        h, w, ch = features.shape
        features_reduces = pca.fit_transform(features.reshape(-1, ch))
        features_reduced_normalized = (features_reduces - np.amin(features_reduces)) / (
                    np.amax(features_reduces) - np.amin(features_reduces))
        output.append(features_reduced_normalized)
    return output


if __name__ == "__main__":
    im1 = np.random.rand(300, 300, 3)
    im2 = np.random.rand(300, 300, 3)
    f1 = np.random.rand(300, 300, 32)
    f2 = np.random.rand(300, 300, 32)
    fig = visualize_closest_points(im1, im2, f1, f2, number_of_correspondences=5)
    print(fig)
    plt.show()
