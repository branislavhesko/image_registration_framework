import numpy as np
import torch
import torch.nn.functional as F

from config import LossConfiguration


class HardestContrastiveLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._positive = PositiveHardestContrastiveLoss(config)
        self._negative = NegativeHardestContrastiveLoss(config)

    def forward(self, inputs):
        feats1 = inputs[0]
        feats2 = inputs[1]
        pos_pairs = inputs[2]

        positive_loss = self._positive(feats1, feats2, pos_pairs)
        negative_loss = self._negative(feats1, feats2, pos_pairs)
        return positive_loss + negative_loss


class PositiveHardestContrastiveLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config

    def forward(self, feats1, feats2, positive_pairs):
        feats1_selected = feats1[0, :, 0]
        feats2_selected = feats2[0, :, 1]
        return torch.mean(torch.sqrt(torch.sum(torch.pow(feats1_selected - feats2_selected, 2), dim=0)))


class NegativeHardestContrastiveLoss(torch.nn.Module):
    PIXEL_LIMIT = 10
    NUM_NEG_PAIRS = 64
    NUM_NEG_INDICES_FOR_LOSS = 5

    def __init__(self, config):
        super().__init__()
        self._config = config

    def forward(self, feats1, feats2, positive_pairs):
        B, C, H, W = feats1.shape
        feats1_flat = feats1.view(C, -1)
        feats2_flat = feats2.view(C, -1)
        negative_loss = 0
        negative_indices1 = positive_pairs[0, :, 0].cpu().numpy()
        for index, negative_idx in enumerate(negative_indices1):
            mask = np.array([np.arange(negative_idx + W * (i - self.PIXEL_LIMIT) - self.PIXEL_LIMIT,
                                       negative_idx + W * (i - self.PIXEL_LIMIT) + self.PIXEL_LIMIT)
                             for i in range(self.PIXEL_LIMIT * 2)]).reshape(1, -1)
            mask = mask[(mask > 0) & (mask < feats1_flat.shape[-1])]
            dist = F.relu(torch.sum(torch.pow(feats1_flat[
                                              :, negative_idx].unsqueeze(dim=1) - feats2_flat, 2), dim=0))
            dist_sorted_indices = dist.argsort()
            negative_indices2 = []

            idx = 0
            counter = 0
            while counter < self.NUM_NEG_INDICES_FOR_LOSS and idx < dist.shape[-1]:
                if not dist[dist_sorted_indices[idx]].item() in mask:
                    counter += 1
                    negative_indices2.append(dist_sorted_indices[idx])
                    mask_new = np.array([np.arange(dist_sorted_indices[idx].item() + W * (
                            i - self.PIXEL_LIMIT) - self.PIXEL_LIMIT,
                                                   dist_sorted_indices[idx].item() + W * (
                                                           i - self.PIXEL_LIMIT) + self.PIXEL_LIMIT)
                                         for i in range(self.PIXEL_LIMIT * 2)]).reshape(1, -1)
                    mask_new = mask_new[(mask_new > 0) & (mask_new < feats1_flat.shape[-1])]
                    mask = np.concatenate([mask, mask_new], axis=0)
                idx += 1

            negative_loss += -torch.mean(dist[negative_indices2, ])

        return negative_loss / self.NUM_NEG_PAIRS


class GeneralVesselLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config

    @staticmethod
    def make_flat(tensor: torch.Tensor):
        return tensor.view(tensor.shape[:2], -1)

    @staticmethod
    def distance(tensor1: torch.Tensor, tensor2: torch.Tensor, dim=0):
        return torch.sqrt(torch.sum(torch.pow(tensor1 - tensor2, 2), dim=dim))

    @staticmethod
    def get_random_choices(features_flat, vessel_mask, num_pairs):
        random_choice10 = np.random.choice(features_flat.shape[-1],
                                           num_pairs, replace=False, p=1 - vessel_mask)
        random_choice21 = np.random.choice(features_flat.shape[-1],
                                           num_pairs, replace=False, p=vessel_mask)
        random_choice11 = np.random.choice(features_flat.shape[-1],
                                           num_pairs, replace=False, p=vessel_mask)
        random_choice20 = np.random.choice(features_flat.shape[-1],
                                           num_pairs, replace=False, p=1 - vessel_mask)
        return random_choice10, random_choice21, random_choice11, random_choice20


class PositiveVesselLoss(GeneralVesselLoss):
    def forward(self, features_flat, vessel_mask_flat):
        rc10, rc21, rc11, rc20 = self.get_random_choices(features_flat, vessel_mask_flat, self._config.NUM_POS_PAIRS)
        return self.distance(features_flat[..., rc10], features_flat[..., rc20], dim=3) + \
            self.distance(features_flat[..., rc11], features_flat[..., rc21], dim=3)


class NegativeVesselLoss(GeneralVesselLoss):

    def forward(self, features_flat, vessel_mask_flat):
        rc10, rc21, _, _ = self.get_random_choices(features_flat, vessel_mask_flat, self._config.NUM_NEG_PAIRS)
        return self.distance(features_flat[..., rc10], features_flat[..., rc21], dim=3)


class TotalVesselLoss(torch.nn.Module):

    def __init__(self, config: LossConfiguration):
        super().__init__()
        self._config = config
        self._neg_loss = NegativeVesselLoss(self._config)
        self._pos_loss = PositiveVesselLoss(self._config)

    def forward(self, features, vessel_mask):
        features_flat, vessel_mask_flat = [
            self.make_flat(t) for t in [features, vessel_mask]]
        positive_loss = self._pos_loss(features_flat, vessel_mask_flat)
        negative_loss = self._neg_loss(features_flat, vessel_mask_flat)
        return self._config.NEGATIVE_LOSS_WEIGHT * negative_loss + positive_loss
