import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from config import ArteryVeinConfiguration, LossConfiguration


class HardestContrastiveLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._positive = PositiveHardestContrastiveLoss(config)
        self._negative = NegativeContrastiveLoss(config)

    def forward(self, inputs):
        feats1 = inputs[0]
        feats2 = inputs[1]
        pos_pairs = inputs[2]

        positive_loss = self._positive(self.make_flat(feats1), self.make_flat(feats2), pos_pairs)
        negative_loss = self._negative(self.make_flat(feats1), self.make_flat(feats2), pos_pairs)
        return positive_loss + torch.nn.functional.relu(self._config.NEGATIVE_LOSS_COEF - negative_loss)

    @staticmethod
    def make_flat(tensor: torch.Tensor):
        return tensor.view(tensor.shape[-3], -1)


class NegativeContrastiveLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config

    def forward(self, feats1, feats2, pos_pairs):
        random_choice1 = np.random.choice(feats1.shape[-1], self._config.NUM_NEG_PAIRS)
        random_choice2 = np.random.choice(feats2.shape[-1], self._config.NUM_NEG_PAIRS)
        feats1_selected = feats1[:, random_choice1]
        feats2_selected = feats2[:, random_choice2]
        return torch.mean(torch.sqrt(torch.sum(torch.pow(feats1_selected - feats2_selected, 2), dim=0) + 1e-7))


class PositiveHardestContrastiveLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config

    def forward(self, feats1, feats2, positive_pairs):
        feats1_selected = feats1[:, positive_pairs[0, :, 0]]
        feats2_selected = feats2[:, positive_pairs[0, :, 1]]
        return torch.mean(torch.sqrt(torch.sum(torch.pow(feats1_selected - feats2_selected, 2), dim=0) + 1e-7))


class NegativeHardestContrastiveLoss(torch.nn.Module):
    PIXEL_LIMIT = 10
    NUM_NEG_PAIRS = 64
    NUM_NEG_INDICES_FOR_LOSS = 5

    def __init__(self, config):
        super().__init__()
        self._config = config

    def forward(self, feats1, feats2, positive_pairs):
        b, c, h, w = feats1.shape
        feats1_flat = feats1.view(c, -1)
        feats2_flat = feats2.view(c, -1)
        negative_loss = 0
        negative_indices1 = positive_pairs[0, :, 0].cpu().numpy()
        for index, negative_idx in enumerate(negative_indices1):
            mask = np.array([np.arange(negative_idx + w * (i - self.PIXEL_LIMIT) - self.PIXEL_LIMIT,
                                       negative_idx + w * (i - self.PIXEL_LIMIT) + self.PIXEL_LIMIT)
                             for i in range(self.PIXEL_LIMIT * 2)]).reshape(1, -1)
            mask = mask[(mask > 0) & (mask < feats1_flat.shape[-1])]
            dist = functional.relu(torch.sum(torch.pow(feats1_flat[
                                              :, negative_idx].unsqueeze(dim=1) - feats2_flat, 2), dim=0))
            dist_sorted_indices = dist.argsort()
            negative_indices2 = []

            idx = 0
            counter = 0
            while counter < self.NUM_NEG_INDICES_FOR_LOSS and idx < dist.shape[-1]:
                if not dist[dist_sorted_indices[idx]].item() in mask:
                    counter += 1
                    negative_indices2.append(dist_sorted_indices[idx])
                    mask_new = np.array([np.arange(dist_sorted_indices[idx].item() + w * (
                            i - self.PIXEL_LIMIT) - self.PIXEL_LIMIT,
                                                   dist_sorted_indices[idx].item() + w * (
                                                           i - self.PIXEL_LIMIT) + self.PIXEL_LIMIT)
                                         for i in range(self.PIXEL_LIMIT * 2)]).reshape(1, -1)
                    mask_new = mask_new[(mask_new > 0) & (mask_new < feats1_flat.shape[-1])]
                    mask = np.concatenate([mask, mask_new], axis=0)
                idx += 1

            negative_loss += torch.mean(dist[negative_indices2, ])

        return negative_loss / self.NUM_NEG_PAIRS


class GeneralVesselLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._config = config

    @staticmethod
    def make_flat(tensor: torch.Tensor):
        return tensor.view(tensor.shape[-3], -1)

    @staticmethod
    def l2_loss(tensor1: torch.Tensor, tensor2: torch.Tensor, dim=0):
        return torch.sqrt(torch.sum(torch.pow(tensor1 - tensor2, 2), dim=dim) + 1e-7)

    @staticmethod
    def distance(tensor1: torch.Tensor, tensor2: torch.Tensor, dim=0):
        return torch.nn.functional.smooth_l1_loss(tensor1, tensor2)

    @staticmethod
    def get_random_choices(features_flat, vessel_mask, num_pairs):
        mask = vessel_mask.squeeze().cpu().numpy()
        random_choice10 = np.random.choice(features_flat.shape[-1],
                                           num_pairs, replace=False, p=(1 - mask) / np.sum(1 - mask))
        random_choice21 = np.random.choice(features_flat.shape[-1],
                                           num_pairs, replace=False, p=mask / np.sum(mask))
        random_choice11 = np.random.choice(features_flat.shape[-1],
                                           num_pairs, replace=False, p=mask / np.sum(mask))
        random_choice20 = np.random.choice(features_flat.shape[-1],
                                           num_pairs, replace=False, p=(1 - mask) / np.sum(1 - mask))
        return random_choice10, random_choice21, random_choice11, random_choice20

    @staticmethod
    def _normalize_probability(mask):
        return mask / np.sum(mask)

    def _get_distances(self, features_flat, artery_choice, background_choice, vein_choice):
        distance_background_vein = self.distance(features_flat[..., background_choice], features_flat[..., vein_choice])
        distance_vein_artery = self.distance(features_flat[..., vein_choice], features_flat[..., artery_choice])
        distance_background_artery = self.distance(
            features_flat[..., background_choice], features_flat[..., artery_choice])
        return distance_background_artery, distance_background_vein, distance_vein_artery


class PositiveVesselLoss(GeneralVesselLoss):
    def forward(self, features_flat, vessel_mask_flat):
        rc10, rc21, rc11, rc20 = self.get_random_choices(features_flat, vessel_mask_flat, self._config.NUM_POS_PAIRS)
        return self.distance(features_flat[..., rc10], features_flat[..., rc20], dim=0) + \
            self.distance(features_flat[..., rc11], features_flat[..., rc21], dim=0)


class NegativeVesselLoss(GeneralVesselLoss):

    def forward(self, features_flat, vessel_mask_flat):
        rc10, rc21, _, _ = self.get_random_choices(features_flat, vessel_mask_flat, self._config.NUM_NEG_PAIRS)
        return self.distance(features_flat[..., rc10], features_flat[..., rc21], dim=0)


class TotalVesselLoss(GeneralVesselLoss):

    def __init__(self, config: LossConfiguration):
        super().__init__(config)
        self._neg_loss = NegativeVesselLoss(self._config)
        self._pos_loss = PositiveVesselLoss(self._config)

    def forward(self, features, vessel_mask):
        print("Mean feats: {}".format(torch.mean(features)))
        features_flat, vessel_mask_flat = [
            self.make_flat(t) for t in [features, vessel_mask]]
        positive_loss = self._pos_loss(features_flat, vessel_mask_flat)
        print("Positive loss: {}".format(torch.max(positive_loss)))
        negative_loss = self._neg_loss(features_flat, vessel_mask_flat)
        print("Negative loss: {}".format(torch.max(negative_loss)))
        return self._config.NEGATIVE_LOSS_WEIGHT * torch.mean(negative_loss) + torch.mean(positive_loss)


class HardestNegativeVesselLoss(GeneralVesselLoss):

    def forward(self, features_flat, vessel_mask_flat):
        pass


class HardestPositiveVesseloss(GeneralVesselLoss):

    def forward(self, features_flat, vessel_mask_flat):
        pass

    @staticmethod
    def get_random_hardest_choices(features_flat, mask, num_pairs):
        random_choice = np.random.choice(features_flat.shape[-1], num_pairs, p=mask / np.sum(mask))
        indices = []
        for choice in random_choice:
            indices.append(torch.argmax(torch.abs(features_flat - features_flat[choice]) * (-1) * mask))
        return random_choice, indices


class ArteryVeinLoss(GeneralVesselLoss):

    def __init__(self, config: ArteryVeinConfiguration):
        super().__init__(config)
        self._positive_loss = PositiveArteryVeinLoss(config)
        self._negative_loss = NegativeArteryVeinLoss(config)
        self._negative_loss2 = NearVesselBackgroundLoss(config)

    def forward(self, features, mask):
        features_flat, mask_flat = self.make_flat(features), self.make_flat(mask).squeeze().cpu().numpy()
        positive_loss = self._positive_loss(features_flat, mask_flat)
        negative_loss = self._negative_loss(features_flat, mask_flat)
        negative_loss2 = self._negative_loss2(features_flat, mask)
        return positive_loss.mean() + self._config.NEGATIVE_LOSS_COEF * functional.relu(
            self._config.NEG_LOSS_CONSTANT - negative_loss.mean()) + self._config.NEGATIVE_LOSS_COEF * functional.relu(
            self._config.NEG_LOSS_CONSTANT - negative_loss2.mean())


class PositiveArteryVeinLoss(GeneralVesselLoss):

    def __init__(self, config: ArteryVeinConfiguration):
        super().__init__(config)

    def forward(self, features_flat, mask_flat):
        vein_mask = self._normalize_probability(mask_flat == 1)
        artery_mask = self._normalize_probability(mask_flat == 2)
        background_mask = self._normalize_probability(mask_flat == 0)
        background_choice1 = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=background_mask)
        background_choice2 = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=background_mask)

        vein_choice1 = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=vein_mask)
        vein_choice2 = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=vein_mask)
        artery_choice1 = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=artery_mask)
        artery_choice2 = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=artery_mask)

        distance_vein = self.distance(features_flat[..., vein_choice1], features_flat[..., artery_choice2])
        distance_artery = self.distance(features_flat[..., artery_choice1], features_flat[..., vein_choice2])
        distance_background = self.distance(
            features_flat[..., background_choice1], features_flat[..., background_choice2])

        self._logger.info(
            "Positive artery-vein loss, distance vein, min, max, mean: {:.3f}, {:.3f}, {:.3f}".format(
                distance_vein.min().item(), distance_vein.max().item(), distance_vein.mean().item()))
        self._logger.info(
            "Positive artery-vein loss, distance artery: min, max, mean: {:.3f}, {:.3f}, {:.3f}".format(
                distance_artery.min().item(), distance_artery.max().item(), distance_artery.mean().item()))
        self._logger.info(
            "Positive artery-vein loss, distance background: min, max, mean: {:.3f}, {:.3f}, {:.3f}".format(
                distance_background.min().item(), distance_background.max().item(), distance_background.mean().item()))
        return distance_artery + distance_background + distance_vein


class NegativeArteryVeinLoss(GeneralVesselLoss):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, features_flat, mask_flat):
        vein_mask = self._normalize_probability(mask_flat == 1)
        artery_mask = self._normalize_probability(mask_flat == 2)
        background_mask = self._normalize_probability(mask_flat == 0)
        background_choice = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=background_mask)
        vein_choice = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=vein_mask)
        artery_choice = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=artery_mask)

        distance_background_artery, distance_background_vein, distance_vein_artery = self._get_distances(features_flat,
                                                                                                         artery_choice,
                                                                                                         background_choice,
                                                                                                         vein_choice)

        self._logger.info(
            "Negative artery-vein loss, distance vein-background: min, max, mean: {:.3f}, {:.3f}, {:.3f}".format(
                distance_background_vein.min().item(), distance_background_vein.max().item(),
                distance_background_vein.mean().item()))
        self._logger.info(
            "Negative artery-vein loss, distance artery-background: min, max, mean: {:.3f}, {:.3f}, {:.3f}".format(
                distance_background_artery.min().item(), distance_background_artery.max().item(),
                distance_background_artery.mean().item()))
        self._logger.info(
            "Negative artery-vein loss, distance vein-artery: min, max, mean: {:.3f}, {:.3f}, {:.3f}".format(
                distance_vein_artery.min().item(), distance_vein_artery.max().item(),
                distance_vein_artery.mean().item()))
        return distance_background_artery + distance_background_vein + distance_vein_artery


class NearVesselBackgroundLoss(GeneralVesselLoss):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, features_flat, mask):
        mask = mask.cpu().squeeze().numpy() if type(mask) == torch.Tensor else mask
        mask_bw = np.zeros_like(mask).astype(np.uint8)
        mask_bw[(mask > 0) & (mask < 2.5)] = 255
        mask_enlarged = cv2.dilate(mask_bw, np.ones((11, 11)))
        background_mask = mask_enlarged - mask_bw
        mask_flat = mask.flatten()
        background_mask = background_mask / np.sum(background_mask)
        background_mask = background_mask.flatten()
        vein_mask = self._normalize_probability(mask_flat == 1)
        artery_mask = self._normalize_probability(mask_flat == 2)
        artery_choice, background_choice, vein_choice = self._get_random_choice(artery_mask, background_mask,
                                                                                features_flat, vein_mask)
        distance_background_artery, distance_background_vein, \
            distance_vein_artery = self._get_distances(features_flat, artery_choice, background_choice, vein_choice)
        return distance_background_artery + distance_background_vein

    def _get_random_choice(self, artery_mask, background_mask, features_flat, vein_mask):
        background_choice = np.random.choice(features_flat.shape[-1], self._config.NUM_NEG_PAIRS, p=background_mask)
        vein_choice = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=vein_mask)
        artery_choice = np.random.choice(features_flat.shape[-1], self._config.NUM_POS_PAIRS, p=artery_mask)
        return artery_choice, background_choice, vein_choice
