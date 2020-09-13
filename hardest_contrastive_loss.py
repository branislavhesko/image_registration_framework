import numpy as np
import torch

from loss import GeneralVesselLoss


class HardestContrastiveLoss(torch.nn.Module):

    def __init__(self, num_classes):
        super(HardestContrastiveLoss, self).__init__()
        self._num_classes = num_classes

    def forward(self, feature_map, mask):
        for class_id in range(self._num_classes):
            pass


class HardestPositiveContrastiveLoss(torch.nn.Module):

    def __init__(self, num_classes=2, num_pairs=1000, to_pick=100):
        super(HardestPositiveContrastiveLoss, self).__init__()
        self._num_classes = num_classes
        self._num_pairs = num_pairs
        self._to_pick = to_pick

    def _unravel_indices(self, indices):
        return indices // self._num_pairs, indices % self._num_pairs

    def _distance(self, feats1, feats2):
        return GeneralVesselLoss.l2_loss(feats1, feats2, dim=-1)

    def forward(self, feature_map, mask):
        picked_ids1 = np.random.choice(
            feature_map.shape[0], self._num_pairs, replace=False, p=(1 - mask) / np.sum(1 - mask))
        picked_ids2 = np.random.choice(feature_map.shape[0], self._num_pairs, replace=False, p=mask / np.sum(mask))
        choice1 = feature_map[picked_ids1, :]
        choice2 = feature_map[picked_ids2, :]
        distance = torch.sqrt((choice1.unsqueeze(1) - choice2.unsqueeze(0)).pow(2).sum(2))
        return distance.view(-1)[torch.argsort(distance.view(-1), descending=True)[:self._to_pick]].mean()


class HardestNegativeContrastiveLoss(torch.nn.Module):
    pass


if __name__ == "__main__":
    feats = torch.rand(1000, 10)
    mask = (torch.rand(1000) > 0.5).int()
    pos_loss = HardestPositiveContrastiveLoss(num_pairs=100, to_pick=10)
    print(pos_loss(feats, mask.numpy()))