import numpy as np
from matplotlib import pyplot as plt

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
    fig = plt.figure(1, dpi=200, figsize=(16, 9))
    plt.subplot(1, 2, 1)
    image1[pairs1[:, 0], pairs1[:, 1], :] = np.array((255, 0, 0))
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    image2[pairs2[:, 0], pairs2[:, 1], :] = np.array((0, 255, 0))
    plt.imshow(image2)
    return fig


if __name__ == "__main__":
    from dataloaders.dataset import RetinaDatasetPop2
    from config import Configuration

    data_loader = RetinaDatasetPop2(Configuration())
    for data in data_loader:
        fig = visualize_points(data[0].permute([1, 2, 0]).numpy(), data[1].permute([1, 2, 0]).numpy(), data[2])
        fig.savefig("skuska.png")
        exit(-1)